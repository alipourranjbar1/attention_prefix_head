"""Held-out layer scores. Arrays are [prompt, layer, offset, hidden]."""
import csv
from pathlib import Path
import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-20)


def grams(x):
    x = x - x.mean(axis=1, keepdims=True)
    k = x @ x.transpose(0, 2, 1)
    return unit(k.reshape(len(k), -1))


def ugrams(x):
    # U-centering removes the positive diagonal bias in small-sample CKA.
    n = x.shape[1]
    k = x @ x.transpose(0, 2, 1)
    idx = np.arange(n)
    k[:, idx, idx] = 0
    sums = k.sum(axis=2, keepdims=True)
    k = k - (sums + sums.transpose(0, 2, 1)) / (n - 2) + sums.sum(axis=1, keepdims=True) / ((n-1)*(n-2))
    k[:, idx, idx] = 0
    return unit(k.reshape(len(k), -1))


def score_slot(draft, target, nfit, ridge):
    # [layer, prompt, width]. Fit a separate centered affine probe for each
    # draft layer and slot. Reuse its dual solve across all target layers.
    d, t = draft[:, nfit:], target[:, nfit:]
    n = d.shape[1]
    dot = lambda a, b: a.reshape(len(a), -1) @ b.reshape(len(b), -1).T
    de = np.einsum('ind,ind->i', d, d)[:, None]
    te = np.einsum('ind,ind->i', t, t)[None, :]
    scores = {
        'cosine': dot(unit(d), unit(t)) / n,
        'relative_mse': np.maximum(de + te - 2*dot(d, t), 0) / np.maximum(te, 1e-20),
        'linear_cka': grams(d) @ grams(t).T,
        'debiased_cka': ugrams(d) @ ugrams(t).T,
    }
    y = target[:, :nfit]
    ymean = y.mean(axis=1, keepdims=True)
    y = y - ymean
    truth = t - ymean
    baseline = (truth**2).sum(axis=(1, 2))
    errs = []
    for x, query in zip(draft[:, :nfit], d):
        mean = x.mean(axis=0)
        x = x - mean
        k = (x @ x.T).astype(np.float64)
        k.flat[::nfit+1] += ridge * max(k.trace()/nfit, 1e-12)
        coeff = np.linalg.solve(k, ((query - mean) @ x.T).T).T.astype(np.float32)
        errs.append(np.square(coeff @ y - truth).sum(axis=(1, 2)))
    errs = np.stack(errs)
    scores['probe_relative_mse'] = errs / np.maximum(te, 1e-20)
    scores['probe_r2'] = 1 - errs / np.maximum(baseline, 1e-20)
    return scores


def heatmap(path, name, matrix):
    # Portable standalone image on training nodes without matplotlib.
    lo, hi = matrix.min(), matrix.max()
    pixels = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            z = (val-lo)/max(hi-lo, 1e-12)
            color = f'rgb({int(240-205*z)},{int(246-133*z)},{int(248-89*z)})'
            pixels.append(f'<rect x="{62+22*j}" y="{65+28*i}" width="21" height="27" fill="{color}"><title>D{i} T{j}: {val:.5f}</title></rect>')
    labels = [f'<text x="{62+22*j}" y="56">{j}</text>' for j in range(matrix.shape[1])]
    labels += [f'<text x="22" y="{83+28*i}">D{i}</text>' for i in range(len(matrix))]
    title = f'{name} | target blocks (zero-based) | light {lo:.3g} to dark {hi:.3g}'
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(580,70+22*matrix.shape[1])}" height="{75+28*len(matrix)}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="11"><text x="12" y="24">{title}</text>{"".join(labels+pixels)}</g></svg>')


def report(draft, target, nfit, ridge, out):
    nslots = draft.shape[2]
    fields = None
    totals = {}
    with (out/'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        for s in range(nslots):
            one = lambda a: np.ascontiguousarray(a[:, :, s].transpose(1, 0, 2), dtype=np.float32)
            scores = score_slot(one(draft), one(target), nfit, ridge)
            if fields is None:
                fields = list(scores)
                writer.writerow(['slot', 'draft_block', 'target_block', *fields])
            for key, arr in scores.items():
                if not np.isfinite(arr).all():
                    raise ValueError(f'Nonfinite {key} at slot {s+1}')
                totals[key] = totals.get(key, 0) + arr.astype(np.float64)
            for i, j in np.ndindex(scores[fields[0]].shape):
                writer.writerow([s+1, i, j, *(scores[k][i,j] for k in fields)])
    with (out/'best_layers.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric','draft_block','best_target_block','mean_over_slots','penultimate_target_score'])
        for key, total in totals.items():
            avg = total / nslots
            heatmap(out/f'{key}.svg', key, avg)
            for i, row in enumerate(avg):
                j = int(np.argmin(row) if 'mse' in key else np.argmax(row))
                writer.writerow([key,i,j,row[j],row[-2]])
