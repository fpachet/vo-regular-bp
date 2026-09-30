#!/usr/bin/env python3
"""MaxOrder reconstruction using an explicitly selected vo_regular_bp checkout.

Input: JSON array of nonempty token arrays. No implicit tokenization, sentence
joining, padding, end symbols, case folding, or corpus download is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


def load_library(path):
    sys.path.insert(0, str(path.resolve()))
    global ContextGraph, Edge, max_order_acceptor, run_bp, true_acceptor
    from vo_regular_bp import ContextGraph, max_order_acceptor, run_bp, true_acceptor
    from vo_regular_bp.context import Edge


def strict_graph(corpus, order):
    """Pooled MLE startup, then strict d-context transitions; no terminal backoff.

    A terminal context without continuations has no outgoing edges. Therefore
    run_bp(..., true_acceptor(), n) conditions on surviving to horizon n.
    """
    counts = defaultdict(Counter)
    for seq in corpus:
        for i, symbol in enumerate(seq):
            for k in range(min(i, order) + 1):
                counts[tuple(seq[i-k:i]) if k else ()][symbol] += 1
    edges = {}
    for state, row in sorted(counts.items()):
        total = sum(row.values())
        edges[state] = [Edge(symbol, count / total, (state + (symbol,))[-order:])
                        for symbol, count in sorted(row.items())]
    return ContextGraph(edges, max_order=order), counts


class CopyIndex:
    """Independent literal-window index, separate from the library acceptor."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.cache = {}

    def windows(self, size):
        if size not in self.cache:
            self.cache[size] = {tuple(s[i:i+size]) for s in self.corpus
                                for i in range(len(s)-size+1)}
        return self.cache[size]

    def longest(self, seq):
        low, high = 0, min(len(seq), max(map(len, self.corpus)))
        while low < high:
            mid = (low + high + 1) // 2
            found = any(tuple(seq[i:i+mid]) in self.windows(mid)
                        for i in range(len(seq)-mid+1))
            if found:
                low = mid
            else:
                high = mid - 1
        return low


def support_count(bp):
    """Integer path counting on the deterministic product (not probability)."""
    counts = {s: int(bp.acceptor.is_accepting(s[1])) for s in bp.layers[-1]}
    for rows in reversed(bp.edges):
        counts = {s: sum(counts.get(e.next_state, 0) for e in row)
                  for s, row in rows.items()}
    return counts.get(bp.start_state, 0)


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1) * p
    a, b = math.floor(x), math.ceil(x)
    return values[a] + (x-a) * (values[b]-values[a])


def statistics(values):
    return {"min": min(values), "q1": quantile(values, .25),
            "median": quantile(values, .5), "q3": quantile(values, .75),
            "max": max(values), "histogram": dict(sorted(Counter(values).items()))}


def self_test():
    """Enumerate all supported toy paths with rational MLE probabilities."""
    corpus = [list("ABRACADABRA")]
    index = CopyIndex(corpus)
    graph, counts = strict_graph(corpus, 1)
    paths = [((), (), Fraction(1))]
    for _ in range(9):
        paths = [(seq+(a,), (a,), p*Fraction(c, sum(counts[state].values())))
                 for seq, state, p in paths for a, c in counts[state].items()]
    exact = {seq: p for seq, _, p in paths if index.longest(seq) < 4}
    bp = run_bp(graph, max_order_acceptor(corpus, 3), length=9)
    mass = sum(exact.values(), Fraction())
    assert support_count(bp) == len(exact)
    assert math.isclose(bp.partition_function, float(mass), rel_tol=1e-12)
    for seq, p in exact.items():
        assert math.isclose(bp.conditional_probability(seq), float(p/mass), rel_tol=1e-12)
    assert index.longest(tuple("RADADACAB")) < 4
    assert index.longest(tuple("ABRADABRACA")) >= 4
    for seq in bp.sample_many(500, rng=17):
        assert seq in exact
    # Check the acceptor against direct substring matching on every length-6
    # word over the alphabet, including unsupported Markov sequences.
    acceptor = max_order_acceptor(corpus, 3)
    for seq in itertools.product(sorted(graph.alphabet), repeat=6):
        state = acceptor.start_state
        for a in seq:
            state = acceptor.next_state(state, a)
            if state is None:
                break
        assert (state is not None and acceptor.is_accepting(state)) == (index.longest(seq) < 4)
    # Strict order-2: no sequence >=3 tokens may have longest copy <3.
    g2, _ = strict_graph(corpus, 2)
    impossible = run_bp(g2, max_order_acceptor(corpus, 2), length=9)
    assert impossible.partition_function == 0 and support_count(impossible) == 0
    return {"status": "passed", "corpus": "ABRACADABRA", "length": 9,
            "forbidden_length": 4, "supported_paths": len(paths),
            "admissible_paths": len(exact), "exact_mass": str(mass),
            "all_conditional_probabilities_checked": True,
            "literal_acceptor_checks": 5**6, "samples_checked": 500,
            "strict_order_2_forbid_trigrams_is_infeasible": True}


def fingerprint(library):
    return {"commit": subprocess.check_output(
                ["git", "-C", str(library), "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "-C", str(library), "status", "--short"], text=True).strip(),
            "python_source_sha256": {str(p.relative_to(library)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((library / "vo_regular_bp").glob("*.py"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--orders", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2014)
    parser.add_argument("--loss-order", type=int, default=3)
    parser.add_argument("--loss-length", type=int, default=20)
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(range(5, 22)))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=["strict_fixed", "longest_suffix_backoff"],
                        default=["strict_fixed", "longest_suffix_backoff"])
    args = parser.parse_args()
    if any(x < 1 for x in args.orders + args.thresholds + [args.length, args.samples, args.loss_order, args.loss_length]):
        parser.error("orders, lengths, thresholds and sample count must be positive")
    load_library(args.library)
    result = {"library": fingerprint(args.library), "seed": args.seed,
              "historical_replication": False, "self_test": self_test()}
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "results.json"

    def save():
        report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.self_test:
        print(json.dumps(result["self_test"], indent=2))
        return
    if args.corpus is None:
        parser.error("--corpus is required unless --self-test is used")
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list) or not corpus or any(
            not isinstance(s, list) or not s or any(not isinstance(t, str) for t in s) for s in corpus):
        parser.error("corpus must be a nonempty JSON list of nonempty lists of string tokens")
    result["corpus"] = {"path": str(args.corpus.resolve()),
                        "sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                        "sequences": len(corpus), "tokens": sum(map(len, corpus)),
                        "vocabulary": len({t for s in corpus for t in s})}
    result["protocol"] = {"startup": "Pooled lower-order MLE until requested context length",
        "fixed_order": "Strict context length thereafter; dead ends terminate; condition on horizon survival",
        "variable_order": "ContextGraph.from_sequences longest known suffix; terminal backoff allowed",
        "boundaries": "No transitions or copied windows across supplied sequence boundaries",
        "quantiles": "Linear interpolation at (sample_count-1)*p",
        "baseline_length": args.length, "sample_count_per_condition": args.samples,
        "loss_order": args.loss_order, "loss_length": args.loss_length,
        "L_semantics": "Reject every copied substring of length L; library max_order=L-1"}
    index = CopyIndex(corpus)
    result["copy_length_experiment"] = []
    for order in ([] if args.skip_baseline else args.orders):
        for mode in args.modes:
            started = time.perf_counter()
            graph = (strict_graph(corpus, order)[0] if mode == "strict_fixed" else
                     ContextGraph.from_sequences(corpus, max_order=order))
            bp = run_bp(graph, true_acceptor(), length=args.length)
            row = {"mode": mode, "order": order, "survival_mass": bp.partition_function}
            if bp.partition_function > 0:
                samples = bp.sample_many(args.samples, rng=random.Random(args.seed + order))
                copies = [index.longest(s) for s in samples]
                if mode == "strict_fixed" and args.length >= order+1:
                    assert min(copies) >= order+1
                row.update(statistics(copies))
                filename = f"samples_{mode}_d{order}.json"
                (args.output / filename).write_text(json.dumps(samples) + "\n")
                row["samples"] = filename
            result["copy_length_experiment"].append(row)
            row["seconds"] = time.perf_counter() - started
            save()
            print(mode, order, row.get("median", "infeasible"), flush=True)
            del bp, graph
    if args.skip_loss:
        return
    graph, _ = strict_graph(corpus, args.loss_order)
    base = run_bp(graph, true_acceptor(), length=args.loss_length)
    total = support_count(base)
    base_mass = base.partition_function
    result["solution_loss"] = {"total_sequences": total, "survival_mass": base_mass, "rows": []}
    del base
    for threshold in args.thresholds:
        started = time.perf_counter()
        # Do not pass alphabet: that would eagerly expand a text-sized dense DFA.
        bp = run_bp(graph, max_order_acceptor(corpus, threshold-1), length=args.loss_length)
        count = support_count(bp)
        row = {"L": threshold, "count": count,
               "surviving_count_fraction": str(Fraction(count, total)) if total else None,
               "solution_loss": float(1-Fraction(count, total)) if total else None,
               "raw_probability_mass": bp.partition_function,
               "conditional_on_survival_mass": bp.partition_function/base_mass if base_mass else None,
               "product_states": bp.time_indexed_product_state_count,
               "product_edges": bp.product_edge_count,
               "seconds": time.perf_counter()-started}
        if count:
            samples = bp.sample_many(min(args.samples, 100), rng=args.seed+threshold)
            assert all(index.longest(s) < threshold for s in samples)
            filename = f"constrained_samples_L{threshold}.json"
            (args.output / filename).write_text(json.dumps(samples) + "\n")
            row["samples"] = filename
        result["solution_loss"]["rows"].append(row)
        save()
        print("L", threshold, "count", count, "mass", bp.partition_function, flush=True)
        del bp


if __name__ == "__main__":
    main()
