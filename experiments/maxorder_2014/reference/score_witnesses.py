#!/usr/bin/env python3
"""Enumerate the L=9 witnesses and rescore at orders 1,2,3 without the library."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('corpus', type=Path)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
corpus = [tuple(s) for s in json.loads(args.corpus.read_text())]
counts = defaultdict(Counter)
for seq in corpus:
    for i, word in enumerate(seq):
        for k in range(min(i,3)+1):
            counts[seq[i-k:i] if k else ()][word] += 1
forbidden = {seq[i:i+9] for seq in corpus for i in range(len(seq)-8)}
witnesses = []
def visit(seq):
    if len(seq) == 20:
        witnesses.append(seq)
        return
    for word in sorted(counts.get(seq[-3:], {})):
        nxt = seq + (word,)
        if len(nxt) < 9 or nxt[-9:] not in forbidden:
            visit(nxt)
visit(())
assert len(witnesses) == 29
records = []
totals = {d: Fraction() for d in [1,2,3]}
for seq in witnesses:
    scores = {}
    for d in [1,2,3]:
        factors = []
        for i, word in enumerate(seq):
            row = counts[seq[max(0,i-d):i]]
            factors.append(Fraction(row[word],sum(row.values())))
        product = Fraction(1)
        for f in factors:
            product *= f
        totals[d] += product
        scores[d] = {'factors': [str(f) for f in factors],
                     'probability': str(product), 'float_probability': float(product)}
    records.append({'sequence': seq, 'scores': scores})
result = {'method': 'Independent strict-order-3 DFS, literal length-9 no-goods, rational rescoring',
    'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
    'count': len(witnesses), 'length':20, 'forbidden_length':9,
    'startup': 'Empirical unigram at position 1; pooled MLE contexts of length min(d,i-1) thereafter',
    'total_mass_by_scoring_order': {d: {'rational': str(v), 'float': float(v)} for d,v in totals.items()},
    'historical_mass': 1e-22, 'relative_difference_order_1': float(totals[1])/1e-22-1,
    'interpretation': 'Order-3 feasibility with order-1 scoring reproduces the published count and quoted probability. This numerical agreement does not establish the historical implementation.',
    'witnesses': records}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({d:float(v) for d,v in totals.items()}))
