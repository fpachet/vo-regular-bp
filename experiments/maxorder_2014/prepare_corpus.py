#!/usr/bin/env python3
"""Extract auditable Johnston corpus variants from downloaded Lib.ru HTML.

The input and tokenized text remain in a supplied analysis directory; the
repository records this recipe and hashes rather than distributing the book.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
args = parser.parse_args()
raw = args.source.read_bytes()
parts = re.findall(r'<h2>Chapter (One|Two|Three|Four|Five|Six|Seven|Eight)</h2></ul>(.*?)(?=<i>Notes to Chapter)', raw.decode('koi8-r'), re.S)
assert len(parts) == 8
word_re = r"[^\W\d_]+(?:[-'][^\W\d_]+)*"
variants = {'chapters_words': [], 'sentences_words': [], 'chapters_with_sentence_markers': []}
bodies = []
for name, body in parts:
    body = re.sub(r'<sup>.*?</sup>', '', body, flags=re.S)
    body = html.unescape(re.sub(r'<[^>]+>', '', body))
    body = re.sub(r'\{[^}]+\}', '', body)
    body = '\n'.join(line.strip() for line in body.splitlines()
                     if line.startswith('     ') and not line.startswith('      '))
    bodies.append(body)
    words = re.findall(word_re, body.lower())
    sentences = [re.findall(word_re, s.lower()) for s in re.split(r'[.!?;]+', body)]
    sentences = [s for s in sentences if s]
    variants['chapters_words'].append(words)
    variants['sentences_words'].extend(sentences)
    variants['chapters_with_sentence_markers'].append([w for s in sentences for w in s + ['<SEP>']])
body_hash = hashlib.sha256(json.dumps(bodies, ensure_ascii=False).encode()).hexdigest()
assert body_hash == '70e14013cc44dcb91c3b306dc94959e94f1695d875d60f82d8c277a16928efe3', 'Extracted source changed; inspect before proceeding'
args.output.mkdir(parents=True, exist_ok=True)
manifest = {'source_url': 'https://lib.ru/LITRA/PUSHKIN/ENGLISH/onegin_j.txt',
    'source_sha256': hashlib.sha256(raw).hexdigest(), 'retrieved': '2026-09-30',
    'extracted_chapter_bodies_sha256': body_hash,
    'source_note': 'Raw hash differs from September 17; all archived chapter token counts, vocabulary counts and segmentation counts agree. HTML includes a mutable popularity footer.',
    'historical_preprocessing_recovered': False,
    'extraction': 'Eight chapter bodies; remove notes, superscripts, page numbers and non-verse indentation; lowercase Unicode words with internal hyphens/apostrophes retained.',
    'word_regex': word_re, 'sentence_split_regex': r'[.!?;]+',
    'primary_variant': 'chapters_with_sentence_markers',
    'rationale': 'The original describes words and sentence separators. Semicolon splitting is an explicit reconstruction assumption, not a recovered historical tokenizer.',
    'variants': {}}
for name, corpus in variants.items():
    p = args.output / (name + '.json')
    p.write_text(json.dumps(corpus, ensure_ascii=False) + '\n')
    manifest['variants'][name] = {'sequences': len(corpus), 'tokens': sum(map(len, corpus)),
        'vocabulary': len({w for s in corpus for w in s}),
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest['variants'], indent=2))
