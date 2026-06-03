"""
EVALUATION SCRIPT: HYBRID ADAPTIVE SEARCH PIPELINE
========================================================================
- Experiments: Evaluate the main Hybrid pipeline over evaluation queries.
- Metrics: Hit@K, MRR@10, route distribution.
- Output: Detailed CSV + summary report.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.controller_retrieval import AdaptiveSearchPipeline

# SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(results: List[Dict], expected_movie_id: int, top_k: int = 10) -> Dict[str, float]:
    retrieved_ids = [r.get("movie_id") for r in results[:top_k]]
    metrics = {
        "hit@1": int(expected_movie_id in retrieved_ids[:1]),
        "hit@3": int(expected_movie_id in retrieved_ids[:3]),
        "hit@5": int(expected_movie_id in retrieved_ids[:5]),
        "hit@10": int(expected_movie_id in retrieved_ids[:top_k]),
        "mrr": 0.0,
    }
    if expected_movie_id in retrieved_ids:
        rank = retrieved_ids.index(expected_movie_id) + 1
        metrics["mrr"] = 1.0 / rank
    return metrics


def evaluate_hybrid(
    eval_file: str,
    output_dir: str = "evaluation",
    sample_size: int = None,
    force_hard: bool = False,
    top_n: int = 10,
) -> None:
    print("\n" + "=" * 100)
    print("EVALUATION: HYBRID ADAPTIVE SEARCH PIPELINE")
    print("=" * 100)

    logger.info(f"Loading evaluation queries from {eval_file}...")
    df = pd.read_csv(eval_file)
    if sample_size:
        df = df.head(sample_size)
        logger.info(f"Using first {sample_size} queries for quick test")

    logger.info(f"Total queries: {len(df)}")

    pipeline = AdaptiveSearchPipeline(
        ci_threshold=0.01,
        min_score_threshold=0.03,
        force_hard=force_hard,
    )

    results_data = []
    agg_metrics = defaultdict(float)
    route_counts = defaultdict(int)

    start_time = time.time()
    for idx, row in df.iterrows():
        query_id = int(row["query_id"])
        query_text = row["query"]
        expected_movie_id = int(row["expected_movie_id"])
        expected_title = row["expected_title"]

        if (idx + 1) % 20 == 0:
            logger.info(f"Progress: {idx + 1}/{len(df)} queries processed...")

        query_start = time.time()
        results, route_name, _ = pipeline.search(query_text, top_n=top_n)
        latency_sec = time.time() - query_start

        route_counts[route_name] += 1
        metrics = calculate_metrics(results, expected_movie_id, top_k=top_n)

        top1_id = results[0].get("movie_id") if results else None
        top1_title = results[0].get("title") if results else "N/A"

        results_data.append({
            "query_id": query_id,
            "query": query_text,
            "expected_movie_id": expected_movie_id,
            "expected_title": expected_title,
            "pred_top1_movie_id": top1_id,
            "pred_top1_title": top1_title,
            "route": route_name,
            "latency_sec": round(latency_sec, 4),
            "hit@1": metrics["hit@1"],
            "hit@3": metrics["hit@3"],
            "hit@5": metrics["hit@5"],
            "hit@10": metrics["hit@10"],
            "mrr": round(metrics["mrr"], 4),
        })

        for k, v in metrics.items():
            agg_metrics[k] += v

    elapsed = time.time() - start_time
    n_queries = len(df)
    avg_latency = elapsed / n_queries if n_queries else 0.0

    final_metrics = {k: agg_metrics[k] / n_queries for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr"]}

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "hybrid_evaluation_results.csv")
    pd.DataFrame(results_data).to_csv(out_csv, index=False)
    logger.info(f"✓ Detailed hybrid results saved to {out_csv}")

    print("\n" + "=" * 100)
    print("SUMMARY: HYBRID PIPELINE")
    print("=" * 100)
    print(f"Queries evaluated: {n_queries}")
    print(f"Force hard: {force_hard}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg latency: {avg_latency:.4f}s")
    print("\nRoute distribution:")
    for route, count in route_counts.items():
        print(f"  {route}: {count} ({count / n_queries:.2%})")

    print("\nMetrics:")
    print(f"  hit@1: {final_metrics['hit@1']:.4f}")
    print(f"  hit@3: {final_metrics['hit@3']:.4f}")
    print(f"  hit@5: {final_metrics['hit@5']:.4f}")
    print(f"  hit@10: {final_metrics['hit@10']:.4f}")
    print(f"  mrr: {final_metrics['mrr']:.4f}")

    print(f"\nDetailed per-query results written to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the hybrid adaptive search pipeline.")
    parser.add_argument("--eval-file", default="evaluation/eval_queries_200.csv", help="Evaluation CSV file path")
    parser.add_argument("--output-dir", default="evaluation", help="Directory to save evaluation output")
    parser.add_argument("--sample", type=int, default=None, help="Run only on the first N queries")
    parser.add_argument("--force-hard", action="store_true", help="Force the pipeline to always use HARD branch")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N results for metrics")
    args = parser.parse_args()

    evaluate_hybrid(
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        sample_size=args.sample,
        force_hard=args.force_hard,
        top_n=args.top_n,
    )
