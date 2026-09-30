from pathlib import Path
import json
r=Path('research/maxorder_reconstruction_2026-09-30')
load=lambda name:json.loads((r/name).read_text())
b=load('sentences_words_baseline/results.json')['copy_length_experiment']
cell=lambda a:f"{a['median']:g} [{a['q1']:g}, {a['q3']:g}]"
by={(x['order'],x['mode']):x for x in b}
assert len(by)==12
rows=[]
mdrows=[]
for d in range(1,7):
 a=cell(by[d,'strict_fixed']);v=cell(by[d,'longest_suffix_backoff'])
 rows.append(f'    {d} & {a} & {v} \\\\')
 mdrows.append(f'| {d} | {a} | {v} |')
table='''\\begin{table}[htbp]
  \\centering
  \\small
  \\begin{tabular}{ccc}
    \\toprule
    Maximum context & Strict fixed order & Longest-suffix backoff \\\\
    \\midrule
'''+ '\n'.join(rows)+'''
    \\bottomrule
  \\end{tabular}
  \\caption{New copy-length experiment on sentence-separated Johnston text:
  median [lower quartile, upper quartile] of the longest copied fragment in
  1,000 generated sequences per condition, each of length 80.
  Fixed-order samples are conditioned on survival to the horizon.}
  \\label{tab:maxorder-reconstruction-copies}
\\end{table}
'''
Path('revised/maxorder-reconstruction-table.tex').write_text(table)
loss=load('sentences_words_full_loss/results.json')['solution_loss']
curve='\n'.join(f"| {a['L']} | {a['count']:,} | {a['solution_loss']:.8f} | {a['raw_probability_mass']:.9g} |" for a in loss['rows'])
sens=[]
for name,title in [('sentences_words_loss','Separate sentences'),('chapters_words_loss','Chapter word streams'),('primary_loss','Chapter streams with sentence markers')]:
 j=load(name+'/results.json')['solution_loss'];a=next(a for a in j['rows'] if a['L']==9)
 sens.append(f"| {title} | {j['total_sequences']:,} | {a['count']:,} | {a['raw_probability_mass']:.9g} |")
length=[]
for d in range(1,7):
 vals=[]
 for name in ['length20_baseline','length40_baseline','primary_baseline']:
  x=next(a for a in load(name+'/results.json')['copy_length_experiment'] if a['order']==d and a['mode']=='strict_fixed')
  vals.append(f"{x['median']:g}")
 length.append('| '+str(d)+' | '+' | '.join(vals)+' |')
report='''# MaxOrder reconstruction results

The reconstruction recovers the published **29 admissible sequences** at order 3,
length 20 and forbidden-copy length 9 when sentences are separate training
sequences. Scoring those same 29 sequences with an **order-1** model gives
**1.0001778490219114 × 10^-22**, agreeing with the historical value to its stated
precision. Their order-3 mass is **4.8815273940871776 × 10^-5**. This identifies a
numerically supported explanation involving different feasibility and scoring
orders; it does not establish the exact historical code or preprocessing.

The copy-length quartiles are not reproduced. The new experiment instead
provides explicit, reproducible measurements for fixed-order and suffix-backoff
sources. The original manuscript is unchanged; the revised MaxOrder section
now includes these findings separately from the historical material.

## Source and protocol

The [Johnston text](https://lib.ru/LITRA/PUSHKIN/ENGLISH/onegin_j.txt) was
retrieved on 30 September 2026. `prepare_corpus.py` extracts the eight chapter
bodies, removes notes and page numbers, lowercases Unicode word tokens and
retains internal apostrophes and hyphens. Splitting on periods, question marks,
exclamation marks and semicolons yields 2,150 segments, 32,723 words and 6,910
distinct words. The historical counts are 2,160, 32,719 and 6,919. No rules were
adjusted to force these counts to agree. The source hash changed since the
September 17 download, but every previously recorded chapter token count,
vocabulary count and segmentation count agrees; the HTML includes a mutable
popularity footer. Exact source and tokenized-file hashes are in
[the corpus manifest](corpus_manifest.json).

The library checkout was clean at commit
`614697b3e16444dc7ca81d4f74fe273b7d3b28a0` of
[vo-regular-bp](https://github.com/fpachet/vo-regular-bp).
All runs record hashes of its top-level Python modules. No library changes were
needed. L denotes a forbidden copied length; `max_order_acceptor` receives L−1.

Strict order d uses unigram startup and pooled empirical context counts of
length min(d,i−1), without falling back after reaching d. A context with no
continuation terminates the process. BP conditions on survival to the requested
horizon. The comparison source is `ContextGraph.from_sequences`, which uses
the longest observed suffix and backs off at terminal contexts. It is not the
library's constraint-dependent order-stack policy. Neither procedure is claimed
to reproduce the old CSP value-ordering heuristic.

## Recovery of the 29 sequences and their scores

The sentence-separated corpus has 15,384 supported strict-order-3 paths of
length 20. Exactly 29 avoid all copied nine-word substrings. The counts were
checked independently, and all 29 witnesses were separately enumerated using
literal forbidden substrings without importing the library.

| Scoring order | Total raw mass of the same 29 sequences |
|---|---:|
| 1 | 1.0001778490219114 × 10^-22 |
| 2 | 1.474857172126263 × 10^-6 |
| 3 | 4.8815273940871776 × 10^-5 |

[Witness scores](witness_scores.json) contain all sequences, all 20 exact
rational factors per scoring order, their products and totals. The order-1 mass
is about 0.0178% above 10^-22, well within the precision of that historical
one-significant-digit report. This agreement is evidence for mixed feasibility
and scoring conventions, not proof of implementation identity.

The order-3 survival mass at n=20 is 0.22924185062542313. Given survival, the
MaxOrder event has mass 0.00021294224334558794. Consequently the quoted 10^-22
must not be treated as this order-3 generator's rejection acceptance rate.

## Sensitivity to sentence boundaries

The separator-stream protocol was run first because the paper mentions word
and sentence-separator symbols. The independent-sentence result was found in
the planned boundary sensitivity comparison; all alternatives are retained.

| Training and copy boundaries | Total paths at n=20 | Paths with L=9 | Raw order-3 mass |
|---|---:|---:|---:|
'''+ '\n'.join(sens)+'''

## Copy length versus order

For each source and order, 1,000 length-80 sequences were sampled with
`random.Random(2014 + order)`. Length 80 and the exact sample count are new
protocol choices: the historical paper does not specify them sufficiently for
replication. Brackets give linearly interpolated first and third quartiles.
Copied fragments are contiguous matches inside individual training sentences;
no match crosses sentence boundaries.

| Maximum context | Strict fixed order | Longest-suffix backoff |
|---|---:|---:|
'''+ '\n'.join(mdrows)+'''

Strict-order results condition on horizon survival. At orders 4–6 only about
0.264% of initial mass survives to 80 tokens, favoring long available fragments;
the backoff source survives with mass 1. This difference is part of the protocol,
not an exactness failure. Both columns differ from the historical table.
The strict order-2 minimum is at least 3 by construction and was checked on every
sample; the historical median 2 remains incompatible with that convention.

An additional horizon comparison keeps the separator-stream corpus and strict
source fixed. The following entries are medians from 1,000 samples per cell:

| Order | Length 20 | Length 40 | Length 80 |
|---|---:|---:|---:|
'''+ '\n'.join(length)+'''

These horizon results do not use the sentence-separated corpus from the preceding
table. The distinction is intentional: they assess horizon sensitivity within
the initially specified separator-stream protocol.

## Recomputed solution loss curve

For sentence-separated text, strict order 3 and n=20:

| Forbidden length L | Admissible paths | Solution loss | Raw order-3 mass |
|---|---:|---:|---:|
'''+curve+'''

The L=21 event is vacuous at horizon 20, so its raw mass equals the survival mass,
not 1. [Independent checks](independent_full_curve_verification.json) use
literal corpus suffixes and forward DP with arbitrary-precision integers and
rational probabilities for all 17 thresholds. They agree with the library's
backward product DP counts and floating-point masses. An additional independent
check covers the separator-stream variant at L=5,9,21.

## Reproduction

Use Python 3.10 or later and the recorded library revision. The scripts use only
the Python standard library plus `vo_regular_bp`; the full book is fetched
separately into an analysis directory. The extracted chapter-body hash is checked before writing the tokenized corpus.
Changes to the HTML popularity counter therefore do not block reproduction,
while changes to the extracted text require inspection.

```sh
curl -L --fail 'https://lib.ru/LITRA/PUSHKIN/ENGLISH/onegin_j.txt' -o /tmp/onegin.html
python3 prepare_corpus.py /tmp/onegin.html --output /tmp/onegin \\
  --manifest corpus_manifest.json
PYTHONHASHSEED=0 python3 reproduce.py --library /path/to/vo_regular_bp \\
  --corpus /tmp/onegin/sentences_words.json --output sentences_words_full_loss \\
  --skip-baseline
PYTHONHASHSEED=0 python3 reproduce.py --library /path/to/vo_regular_bp \\
  --corpus /tmp/onegin/sentences_words.json --output sentences_words_baseline \\
  --skip-loss --length 80 --samples 1000
python3 verify_counts.py /tmp/onegin/sentences_words.json \\
  sentences_words_full_loss/results.json --output independent_full_curve_verification.json \\
  --thresholds 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
python3 score_witnesses.py /tmp/onegin/sentences_words.json --output witness_scores.json
```

Use the other two prepared corpus variants to rerun the boundary comparisons.
Use `--length 20` or `--length 40 --modes strict_fixed` for the horizon checks
(the latter mode flag applies to both runs). The repository retains the source
recipe, numerical results, generated samples and verification records; corpus
identity with the old experiment remains unclaimed.
'''
(r/'README.md').write_text(report)
print(table)
