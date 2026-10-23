"""Exact greedy acceptance boundaries; no tensor dependency.

Cached predictions are valid only while preceding proposals equal the reference.
A rejected proposal terminates verification before any later context matters.
"""


def cached_boundary(proposals, reference, target_ids, eos_ids=()):
    """Return (length, status): exact, verify, or eos.

    target_ids[j] was evaluated on the *reference* prefix before position j.
    A matched proposal that departs from that reference invalidates later scores.
    An accepted last proposal requires no additional score for its length label.
    """
    if not proposals or not (len(proposals) == len(reference) == len(target_ids)):
        raise ValueError('Expected equally sized, nonempty blocks')
    eos = set(eos_ids)
    for j, (draft, ref, target) in enumerate(zip(proposals, reference, target_ids)):
        if target in eos:
            return None, 'eos'
        if draft != target:
            return j, 'exact'
        if j + 1 < len(proposals) and draft != ref:
            return None, 'verify'
    return len(proposals), 'exact'


def verified_boundary(proposals, target_ids, eos_ids=()):
    """Target scores here are conditioned on the actual proposal prefix."""
    if not proposals or len(proposals) != len(target_ids):
        raise ValueError('Expected equally sized, nonempty blocks')
    eos = set(eos_ids)
    for j, (draft, target) in enumerate(zip(proposals, target_ids)):
        if target in eos:
            return None, 'eos'
        if draft != target:
            return j, 'exact'
    return len(proposals), 'exact'


def eligible_starts(mask, documents, block_size):
    """Complete response blocks only; no padding/document/mask boundary labels."""
    if len(mask) != len(documents) or block_size < 2:
        raise ValueError('Invalid mask/document lengths or block size')
    return [a for a in range(len(mask)-block_size+1)
            if documents[a] >= 0 and all(mask[a:a+block_size])
            and all(d == documents[a] for d in documents[a:a+block_size])]
