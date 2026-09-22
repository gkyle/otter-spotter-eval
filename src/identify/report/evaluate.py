#!/usr/bin/env python3
"""Encounter-aware evaluation of embedding or pairwise-score re-ID backends.

Protocol (closed-set, leave-one-out with same-encounter exclusion):

* Each embedded crop is used as a query in turn.
* Its gallery is every *other* crop whose ``observation_id`` (encounter) differs
  from the query's. Excluding the query's own encounter prevents trivially
  matching near-duplicate photos from the same sighting.
* Gallery items are ranked by cosine similarity (embeddings are L2-normalized,
  so cosine == dot product). A gallery item is a correct match if it shares the
  query's ``individual_id``.
* A query is *evaluable* only if at least one correct match exists in its
  gallery (i.e. the individual was seen in >= 2 encounters). Singletons are
  skipped -- they cannot be scored.

Reported per crop target: Rank-1/5/10 (CMC) and mAP over evaluable queries,
plus a per-individual breakdown and a random-ranking chance baseline. These
mirror common wildlife re-identification metrics so backends are comparable.

Usage:
    uv run src/identify/report/evaluate.py
    uv run src/identify/report/evaluate.py --body-part whole_body --ranks 1 5 10 20
    uv run src/identify/report/evaluate.py --location-weight 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import BODY_PARTS, DATA_DIR, REPO_ROOT  # noqa: E402
from common.similarity_data import load_similarity  # noqa: E402

EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "miewid-msv3-finetuned"
EVAL_DIR = DATA_DIR / "eval" / "miewid-msv3-finetuned"


def average_precision(relevant_sorted: np.ndarray) -> float:
    """Average precision for a gallery ranking. ``relevant_sorted`` is a boolean
    array (gallery ordered by descending similarity)."""
    n_relevant = int(relevant_sorted.sum())
    if n_relevant == 0:
        return float("nan")
    cumulative = np.cumsum(relevant_sorted)
    ranks = np.arange(1, len(relevant_sorted) + 1)
    precision_at_hits = cumulative[relevant_sorted] / ranks[relevant_sorted]
    return float(precision_at_hits.mean())


def evaluate(
    similarity: np.ndarray,
    individuals: np.ndarray,
    encounters: np.ndarray,
    ranks: list[int],
) -> pd.DataFrame:
    """Run the leave-one-out evaluation; returns one row per evaluable query."""
    records: list[dict] = []
    for i in range(len(similarity)):
        # Gallery = different encounter than the query (also excludes self).
        gallery = encounters != encounters[i]
        if not gallery.any():
            continue
        order = np.argsort(-similarity[i][gallery])
        gallery_individuals = individuals[gallery][order]
        relevant = gallery_individuals == individuals[i]
        if not relevant.any():
            continue  # singleton individual across encounters -> not evaluable

        first_hit = int(np.argmax(relevant)) + 1  # 1-based rank of first match
        record = {
            "query_index": i,
            "individual_id": individuals[i],
            "observation_id": encounters[i],
            "gallery_size": int(gallery.sum()),
            "n_relevant": int(relevant.sum()),
            "first_hit_rank": first_hit,
            "ap": average_precision(relevant),
        }
        for k in ranks:
            record[f"rank_{k}"] = bool(relevant[:k].any())
        records.append(record)
    return pd.DataFrame(records)


def chance_rank1(queries: pd.DataFrame) -> float:
    """Expected Rank-1 for random ranking: mean(n_relevant / gallery_size)."""
    if queries.empty:
        return float("nan")
    return float((queries["n_relevant"] / queries["gallery_size"]).mean())


def summarize(queries: pd.DataFrame, ranks: list[int]) -> dict:
    metrics: dict = {
        "evaluable_queries": int(len(queries)),
        "individuals_evaluated": int(queries["individual_id"].nunique()) if len(queries) else 0,
        "mAP": float(queries["ap"].mean()) if len(queries) else float("nan"),
        "chance_rank_1": chance_rank1(queries),
    }
    for k in ranks:
        metrics[f"rank_{k}"] = float(queries[f"rank_{k}"].mean()) if len(queries) else float("nan")
    return metrics


def evaluate_body_part(
    body_part: str, emb_dir: Path, ranks: list[int], location_weight: float = 0.0
):
    try:
        similarity, index, source, locations_available = load_similarity(
            emb_dir, body_part, location_weight
        )
    except FileNotFoundError:
        return None, None, None

    queries = evaluate(
        similarity,
        index["individual_id"].to_numpy(),
        index["observation_id"].to_numpy(),
        ranks,
    )
    metrics = summarize(queries, ranks)
    metrics["body_part"] = body_part
    metrics["n_embeddings"] = int(len(index))
    metrics["n_individuals_total"] = int(index["individual_id"].nunique())
    metrics["score_source"] = source
    metrics["location_weight"] = location_weight
    metrics["locations_available"] = (
        locations_available
    )

    per_individual = pd.DataFrame()
    if not queries.empty:
        per_individual = (
            queries.groupby("individual_id")
            .agg(queries=("ap", "size"), mAP=("ap", "mean"),
                 rank_1=("rank_1", "mean"))
            .sort_values("mAP", ascending=False)
        )
    return metrics, queries, per_individual


def print_report(metrics: dict, per_individual: pd.DataFrame, ranks: list[int]) -> None:
    bp = metrics["body_part"]
    print(f"\n=== {bp} ===")
    print(f"  crops: {metrics['n_embeddings']} | individuals labeled: "
          f"{metrics['n_individuals_total']}")
    print(f"  visual score: {metrics['score_source']}")
    if metrics["location_weight"]:
        print(f"  similarity: visual {1 - metrics['location_weight']:.1%} + "
              f"location {metrics['location_weight']:.1%} "
              f"({metrics['locations_available']} locations)")
    if metrics["evaluable_queries"] == 0:
        print("  No evaluable queries: no individual appears in >= 2 encounters.")
        print("  Label the same otter across different observations to enable scoring.")
        return
    print(f"  evaluable queries: {metrics['evaluable_queries']} across "
          f"{metrics['individuals_evaluated']} individuals")
    rank_str = "  ".join(f"Rank-{k} {metrics[f'rank_{k}']*100:5.1f}%" for k in ranks)
    print(f"  {rank_str}")
    print(f"  mAP {metrics['mAP']*100:5.1f}%   (chance Rank-1 ~ "
          f"{metrics['chance_rank_1']*100:.1f}%)")
    if not per_individual.empty:
        print("  per individual (queries / mAP / Rank-1):")
        for ind, row in per_individual.iterrows():
            print(f"    {ind:<16} {int(row['queries']):>2}  "
                  f"mAP {row['mAP']*100:5.1f}%  Rank-1 {row['rank_1']*100:5.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR,
                        help="embeddings root (default: data/embeddings)")
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR,
                        help="where to write metrics/queries (default: data/eval)")
    parser.add_argument("--body-part", choices=BODY_PARTS,
                        help="only evaluate this target (default: all present)")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10],
                        help="CMC ranks to report (default: 1 5 10)")
    parser.add_argument(
        "--location-weight",
        type=float,
        default=0.0,
        help="location share of blended cosine similarity, 0..1 (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.location_weight <= 1:
        print("error: --location-weight must be between 0 and 1", file=sys.stderr)
        return 2
    parts = [args.body_part] if args.body_part else list(BODY_PARTS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for body_part in parts:
        try:
            metrics, queries, per_individual = evaluate_body_part(
                body_part, args.embeddings_dir, args.ranks, args.location_weight
            )
        except (OSError, ValueError) as exc:
            print(f"\n=== {body_part} ===\n  error: {exc}", file=sys.stderr)
            continue
        if metrics is None:
            print(f"\n=== {body_part} ===\n  No backend scores found.")
            continue
        print_report(metrics, per_individual, args.ranks)

        suffix = (
            ""
            if args.location_weight == 0
            else f"_location_{args.location_weight:g}".replace(".", "p")
        )
        (args.output_dir / f"{body_part}{suffix}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        if queries is not None and not queries.empty:
            queries.to_csv(
                args.output_dir / f"{body_part}{suffix}_queries.csv", index=False
            )
        summary_rows.append(metrics)

    if summary_rows:
        resolved_output = args.output_dir.resolve()
        try:
            display_output = resolved_output.relative_to(REPO_ROOT)
        except ValueError:
            display_output = resolved_output
        print(f"\nWrote metrics to {display_output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
