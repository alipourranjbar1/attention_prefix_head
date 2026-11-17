from pathlib import Path
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):
            h.update(part)
    return h.hexdigest()

def quarter_plan(counts_by_update, fraction=0.25):
    """First completed update crossing each dataset-row threshold, not step quarters."""
    if not 0 < fraction <= 1 or not counts_by_update or min(counts_by_update) < 1:
        raise ValueError('Expected a positive fraction and positive per-update document counts')
    total = sum(counts_by_update)
    thresholds = [min((i+1)*fraction,1.0) for i in range(math.ceil(1/fraction))]
    result, consumed, cursor = [], 0, 0
    for step, count in enumerate(counts_by_update,1):
        consumed += count
        crossed = []
        while cursor < len(thresholds) and consumed >= math.ceil(thresholds[cursor]*total-1e-9):
            crossed.append(thresholds[cursor])
            cursor += 1
        if crossed:
            result.append({'step':step,'scheduled_fractions':crossed,
                           'documents_consumed':consumed,'planned_documents':total,
                           'actual_fraction':consumed/total})
    return result
