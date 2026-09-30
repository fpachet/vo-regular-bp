#!/usr/bin/env python3
"""Reproduce the MaxOrder chapter results without changing archived evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
SOURCE_URL = 'https://lib.ru/LITRA/PUSHKIN/ENGLISH/onegin_j.txt'


def check_reference():
    manifest = json.loads((HERE / 'reference/evidence_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((HERE / 'reference' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Reference hash mismatch: {name}')
    print(f"Verified {len(manifest['sha256'])} archived evidence hashes.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, default=HERE.parents[1])
    p.add_argument('--output', type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--source', type=Path, help='Previously downloaded Johnston HTML')
    source.add_argument('--download', action='store_true', help='Fetch the source from Lib.ru')
    p.add_argument('--full', action='store_true', help='Also rerun boundary and horizon comparisons')
    p.add_argument('--figures', action='store_true', help='Requires reportlab; write PDF curve and LaTeX table')
    p.add_argument('--verify-only', action='store_true', help='Check archived file hashes without running experiments')
    args = p.parse_args()
    check_reference()
    if args.verify_only:
        return
    if args.output is None or not (args.source or args.download):
        p.error('a new --output directory and either --source or --download are required')
    out = args.output.resolve()
    if out.exists():
        p.error('--output must not exist; choose a fresh directory to preserve previous runs')
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')

    def run(script, *options):
        command = [sys.executable, str(HERE / script), *map(str, options)]
        print('+', ' '.join(command), flush=True)
        subprocess.run(command, check=True, env=env)

    source_path = args.source.resolve() if args.source else out / 'onegin.html'
    if args.download:
        with urlopen(SOURCE_URL, timeout=60) as response:
            source_path.write_bytes(response.read())
    run('prepare_corpus.py', source_path, '--output', out / 'corpus', '--manifest', out / 'corpus_manifest.json')
    # Timestamp describes this run; the archived manifest retains the historical date.
    from datetime import datetime, timezone
    manifest_path = out / 'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['retrieved'] = datetime.now(timezone.utc).date().isoformat() if args.download else None
    manifest['source_note'] = 'Downloaded by this runner.' if args.download else 'User-supplied source; retrieval date not asserted.'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    sentences = out / 'corpus/sentences_words.json'

    def experiment(name, corpus, *options):
        run('reproduce.py', '--library', args.library.resolve(), '--corpus', corpus,
            '--output', out / name, *options)

    experiment('sentences_words_full_loss', sentences, '--skip-baseline')
    experiment('sentences_words_baseline', sentences, '--skip-loss', '--length', 80, '--samples', 1000)
    run('verify_counts.py', sentences, out / 'sentences_words_full_loss/results.json',
        '--output', out / 'independent_full_curve_verification.json', '--thresholds', *range(5, 22))
    run('score_witnesses.py', sentences, '--output', out / 'witness_scores.json')
    if args.full:
        markers = out / 'corpus/chapters_with_sentence_markers.json'
        experiment('sentences_words_loss', sentences, '--skip-baseline', '--thresholds', 5, 9, 21)
        experiment('chapters_words_loss', out / 'corpus/chapters_words.json', '--skip-baseline', '--thresholds', 5, 9, 21)
        experiment('primary_loss', markers, '--skip-baseline')
        experiment('primary_baseline', markers, '--skip-loss')
        for length in (20, 40):
            experiment(f'length{length}_baseline', markers, '--skip-loss', '--length', length, '--modes', 'strict_fixed')
        for corpus, folder, filename in [(sentences, 'sentences_words_loss', 'independent_sentence_verification.json'),
                                         (markers, 'primary_loss', 'independent_verification.json')]:
            run('verify_counts.py', corpus, out / folder / 'results.json', '--output', out / filename)
    # Compare scientific outputs while allowing paths, timings and source fingerprints to differ.
    checks = []
    for result_path in sorted(out.glob('*/results.json')):
        current = json.loads(result_path.read_text())
        expected = json.loads((HERE / 'reference' / result_path.relative_to(out)).read_text())
        if 'solution_loss' in current:
            a, b = current['solution_loss'], expected['solution_loss']
            assert a['total_sequences'] == b['total_sequences']
            assert [(r['L'], r['count']) for r in a['rows']] == [(r['L'], r['count']) for r in b['rows']]
            from math import isclose
            assert isclose(a['survival_mass'], b['survival_mass'], rel_tol=1e-12)
            for x, y in zip(a['rows'], b['rows']):
                assert isclose(x['raw_probability_mass'], y['raw_probability_mass'], rel_tol=1e-12)
        assert len(current['copy_length_experiment']) == len(expected['copy_length_experiment'])
        for x, y in zip(current['copy_length_experiment'], expected['copy_length_experiment']):
            for key in ('mode', 'order', 'min', 'q1', 'median', 'q3', 'max', 'histogram'):
                assert x[key] == y[key], (result_path.parent.name, key)
            assert (result_path.parent / x['samples']).read_bytes() == (HERE / 'reference' / result_path.parent.name / y['samples']).read_bytes()
        checks.append(result_path.parent.name)
    scores = json.loads((out / 'witness_scores.json').read_text())
    expected_scores = json.loads((HERE / 'reference/witness_scores.json').read_text())
    assert scores == expected_scores
    (out / 'reference_comparison.json').write_text(json.dumps({'status': 'passed', 'experiments': checks,
        'witnesses_and_exact_scores_match': True, 'sample_files_match': True}, indent=2) + '\n')
    if args.figures:
        run('generate_artifacts.py', '--results', out, '--output', out / 'figures')
    print('Reproduction and reference comparison passed:', out, flush=True)


if __name__ == '__main__':
    main()
