"""Measure a model the way the API will run it, on a labelled set of captchas.

Run this on real samples of each customer's captcha before you take them on, and after
every retrain. It uses the server's own Solver, so the numbers match production.

  python -m training.evaluate --model models/current --data datasets/acme-test/labels.txt
  python -m training.evaluate --model runs/acme-v1/model --data ... --failures failures.csv

It prints exact-match accuracy, accuracy by text length, and for several confidence
cut-offs: how many captchas you would answer and how many of those would be right.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from captcha2text.images import decode_image
from captcha2text.model import Solver

from .common import read_labels

THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="folder with model.onnx and model.json")
    p.add_argument("--data", type=Path, nargs="+", required=True, help="labels files")
    p.add_argument("--ignore-case", action="store_true", help="count 'AbC' as matching 'abc'")
    p.add_argument("--failures", type=Path, help="write wrong answers to this CSV")
    args = p.parse_args(argv)

    solver = Solver(args.model)
    items = [it for f in args.data for it in read_labels(f)]
    results = []
    by_len: dict[int, list[bool]] = defaultdict(list)
    total_ms = 0

    for path, expected in items:
        sol = solver.solve(decode_image(path.read_bytes()))
        got = sol.text
        ok = got.lower() == expected.lower() if args.ignore_case else got == expected
        results.append((path, expected, got, sol.confidence, ok))
        by_len[len(expected)].append(ok)
        total_ms += sol.processing_ms

    n = len(results)
    correct = sum(r[4] for r in results)
    print(f"Model {solver.name} on {n} captchas")
    print(f"Exact match: {correct}/{n} = {correct / n:.2%}   average {total_ms / n:.1f} ms per image\n")

    print("By length:")
    for length in sorted(by_len):
        oks = by_len[length]
        print(f"  {length} chars: {sum(oks) / len(oks):.2%} of {len(oks)}")

    print("\nIf you only answer when confidence >= cut-off:")
    print(f"  {'cut-off':>8} {'answered':>9} {'right when answered':>20}")
    for t in THRESHOLDS:
        answered = [r for r in results if r[3] >= t]
        right = sum(r[4] for r in answered)
        share = len(answered) / n
        acc = right / len(answered) if answered else 0.0
        print(f"  {t:>8.2f} {share:>9.1%} {acc:>20.2%}")

    if args.failures:
        with args.failures.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "expected", "got", "confidence"])
            for path, expected, got, conf, ok in results:
                if not ok:
                    w.writerow([path, expected, got, conf])
        print(f"\nWrote {n - correct} failures to {args.failures}")


if __name__ == "__main__":
    main()
