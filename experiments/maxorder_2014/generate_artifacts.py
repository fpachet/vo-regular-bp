"""Render the exact, independently verified curve; no experiment is rerun."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=HERE / 'reference')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = json.loads((args.results / 'independent_full_curve_verification.json').read_text())['verified_rows']
assert [r['L'] for r in rows] == list(range(5, 22))
assert all(r['matches_library'] for r in rows)
total = rows[-1]['count']
survival = Fraction(rows[-1]['rational_mass'])
assert total == 15384
out = args.output / 'maxorder_verified_curve.pdf'
c = canvas.Canvas(str(out), pagesize=(610, 285), invariant=1)
c.setTitle('Verified MaxOrder: solution loss and conditioned probability')
blue, red, grey = map(HexColor, ('#235c88', '#a74329', '#d9d9d9'))
bottom, height, width = 48, 160, 225

def text(x, y, value, size=10, align='left'):
    c.setFillColor(HexColor('#222222'))
    c.setFont('Helvetica', size)
    getattr(c, {'left': 'drawString', 'center': 'drawCentredString', 'right': 'drawRightString'}[align])(x, y, value)

def axes(left, log=False):
    x = lambda L: left + (L-5)/16*width
    y = lambda v: bottom + (v+6)/6*height if log else bottom + v*height
    ticks = list(range(-6, 1)) if log else [0, .25, .5, .75, 1]
    for v in ticks:
        c.setStrokeColor(grey); c.setLineWidth(.4)
        c.line(left, y(v), left+width, y(v))
        text(left-8, y(v)-3, ('1' if v==0 else '10^%d'%v) if log else '%g'%v, align='right')
    c.setStrokeColor(HexColor('#555555')); c.setLineWidth(.7)
    c.line(left, bottom, left, bottom+height)
    c.line(left, bottom, left+width, bottom)
    for L in (5,9,13,17,21):
        c.line(x(L), bottom, x(L), bottom-4)
        text(x(L), bottom-17, str(L), align='center')
    text(left+width/2, 10, 'Forbidden-copy threshold L', 11, 'center')
    return x, y

def curve(points, color, dashed=False):
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.5)
    c.setDash(4, 3) if dashed else c.setDash()
    path=c.beginPath(); path.moveTo(*points[0])
    for xy in points[1:]: path.lineTo(*xy)
    c.drawPath(path)
    c.setDash()
    for xx, yy in points: c.circle(xx, yy, 2, fill=1, stroke=0)

text(48, 270, '(a) Count-based solution loss', 12)
x,y=axes(48)
curve([(x(r['L']), y(1-r['count']/total)) for r in rows], blue)
text(48, 238, 'L = 9: 29 / 15,384 survive', 10)
text(365, 270, '(b) Surviving fraction / probability', 12)
x,y=axes(365, True)
curve([(x(r['L']), y(math.log10(r['count']/total))) for r in rows if r['count']], blue)
curve([(x(r['L']), y(math.log10(float(Fraction(r['rational_mass'])/survival)))) for r in rows if r['count']], red, True)
for yy, label, col, dashed in [(248, 'Count fraction', blue, False), (232, 'Conditional source mass', red, True)]:
    c.setStrokeColor(col); c.setLineWidth(1.5)
    c.setDash(4,3) if dashed else c.setDash()
    c.line(377, yy, 397, yy); c.setDash()
    text(403, yy-3, label, 9)
c.save()
print(out)

load=lambda name:json.loads((args.results/name).read_text())
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
(args.output / 'maxorder-reconstruction-table.tex').write_text(table)
