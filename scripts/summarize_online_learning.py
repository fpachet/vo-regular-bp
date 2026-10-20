"""Summarize raw online-learning JSONL (median of each worker's cycle medians)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

METRICS = ("invalidate_ms", "store_ms", "counts_ms", "graph_ms", "backward_prepare_ms",
           "first_sample_ms", "cached_sample_ms", "learn_to_first_ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = [json.loads(line) for line in (args.directory / "timings.jsonl").read_text().splitlines()]
    summary = {}
    for case in dict.fromkeys(r["case"] for r in runs):
        summary[case] = {}
        for method in ("rebuild", "counts", "incremental"):
            selected = [r for r in runs if r["case"] == case and r["method"] == method]
            summary[case][method] = {}
            for phase in ("cold", "post_learning"):
                worker_medians = [
                    {metric: statistics.median(c.get(metric, 0) for c in r["cycles"] if c["phase"] == phase)
                     for metric in METRICS}
                    for r in selected if any(c["phase"] == phase for c in r["cycles"])
                ]
                if worker_medians:
                    values = {metric: statistics.median(w[metric] for w in worker_medians) for metric in METRICS}
                    values["worker_total_range_ms"] = [min(w["learn_to_first_ms"] for w in worker_medians),
                                                       max(w["learn_to_first_ms"] for w in worker_medians)]
                    summary[case][method][phase] = values
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
