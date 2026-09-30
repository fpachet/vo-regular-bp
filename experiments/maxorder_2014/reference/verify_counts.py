#!/usr/bin/env python3
"""Independent forward DP using literal corpus suffixes, no vo_regular_bp.

Integer counts and rational masses are computed at selected thresholds. This checks the
library's product construction, backward probabilities and support counts.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('corpus', type=Path)
p.add_argument('results', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--thresholds', type=int, nargs='+', default=[5,9,21])
args = p.parse_args()
corpus = [tuple(s) for s in json.loads(args.corpus.read_text())]
result = json.loads(args.results.read_text())
d, n = result['protocol']['loss_order'], result['protocol']['loss_length']
counts = defaultdict(Counter)
for seq in corpus:
    for i, symbol in enumerate(seq):
        for k in range(min(i, d)+1):
            counts[seq[i-k:i] if k else ()][symbol] += 1
factors = {k: {s[i:i+k] for s in corpus for i in range(len(s)-k+1)}
           for k in range(1, n+1)}
checked = []
for L in args.thresholds:
    distribution = {(): (1, Fraction(1))}
    cache = {}
    for t in range(n):
        following = {}
        for state, (number, mass) in distribution.items():
            if state not in cache:
                context = state[-d:] if d else ()
                row = counts.get(context, {})
                total = sum(row.values())
                edges = []
                for symbol, count in row.items():
                    candidate = state + (symbol,)
                    if len(candidate) >= L and candidate[-L:] in factors.get(L, set()):
                        continue
                    # Longest suffix occurring ANYWHERE in a corpus sequence.
                    while candidate and candidate not in factors[len(candidate)]:
                        candidate = candidate[1:]
                    edges.append((candidate, Fraction(count, total)))
                cache[state] = edges
            for target, probability in cache[state]:
                old_number, old_mass = following.get(target, (0, Fraction()))
                following[target] = old_number + number, old_mass + mass*probability
        distribution = following
    count = sum(v[0] for v in distribution.values())
    mass = sum((v[1] for v in distribution.values()), Fraction())
    expected = next(r for r in result['solution_loss']['rows'] if r['L']==L)
    assert count == expected['count'], (L,count,expected['count'])
    assert math.isclose(float(mass), expected['raw_probability_mass'], rel_tol=1e-11, abs_tol=1e-300)
    checked.append({'L': L, 'count': count, 'rational_mass': str(mass),
                    'float_mass': float(mass), 'matches_library': True})
    print(L, count, float(mass), flush=True)
args.output.write_text(json.dumps({'method': 'Independent literal-suffix forward DP with Fraction probabilities',
    'corpus_sha256': result['corpus']['sha256'], 'verified_rows': checked}, indent=2)+'\n')
