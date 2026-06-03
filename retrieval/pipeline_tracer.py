"""
PIPELINE TRACER: Theo dõi toàn bộ luồng xử lý query
========================================================================
- Input: User's raw query
- Output: Detailed trace report (CSV/JSON/TXT) của từng stage
- Giúp debug, analyze bottleneck, verify correctness

Luồng: Query → Encoding → Retrieval → Aggregation → Ranking → Rerank → Final Score
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from query import QueryEncoder
from hybrid import hybrid_retrieval
from bm25 import bm25_retrieval
from dense import dense_retrieval
from aggregate import ChunkAggregator
from rrf import RankFusion
from final_scorer import FinalScorer

# SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineTracer:
    """Tracing tool để theo dõi toàn bộ pipeline"""
    
    def __init__(self, output_dir: str = "trace_logs"):
        """
        Initialize tracer
        
        Args:
            output_dir: Directory để lưu trace logs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info(" Initializing Pipeline Tracer...")
        self.query_encoder = QueryEncoder()
        self.aggregator_1st = ChunkAggregator(max_chunks_per_movie=5)
        self.rank_fusion = RankFusion(rrf_k=60)
        self.final_scorer = FinalScorer(semantic_weight=0.8, popularity_weight=0.2)
        
        # Trace data storage
        self.traces: Dict[str, Any] = {}
        
        logger.info(f" Tracer ready, logs will be saved to: {output_dir}")
    
    def _log_stage(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Log a stage with timestamp"""
        self.traces[stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def _format_vector(self, vector: List[float], max_items: int = 5) -> Dict:
        """Format vector for display (show first few values + stats)"""
        if not vector:
            return {"length": 0, "values": []}
        return {
            "length": len(vector),
            "first_values": vector[:max_items],
            "min": round(min(vector), 4),
            "max": round(max(vector), 4),
            "mean": round(sum(vector) / len(vector), 4)
        }
    
    def _format_sparse_vector(self, sparse_vec: Tuple[List[int], List[float]]) -> Dict:
        """Format sparse vector for display"""
        indices, values = sparse_vec
        return {
            "num_terms": len(indices),
            "first_terms": list(zip(indices[:5], [round(v, 4) for v in values[:5]])),
            "range": (min(indices) if indices else 0, max(indices) if indices else 0)
        }
    
    def _format_chunks(self, chunks: List[Dict], top_k: int = 3) -> Dict:
        """Format chunks summary"""
        return {
            "total_chunks": len(chunks),
            "top_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "score": round(c.get("score", 0), 4),
                    "movie_id": c.get("payload", {}).get("movie_id"),
                    "preview": c.get("payload", {}).get("chunk_text", "")[:100] + "..."
                }
                for c in chunks[:top_k]
            ]
        }
    
    def _format_movies(self, movies: List[Dict], top_k: int = 10) -> Dict:
        """Format movies summary"""
        return {
            "total_movies": len(movies),
            "top_movies": [
                {
                    "movie_id": m.get("movie_id"),
                    "title": m.get("title"),
                    "score": round(m.get("score", 0), 4),
                    "genres": m.get("genres")
                }
                for m in movies[:top_k]
            ]
        }
    
    def trace_query(self, raw_query: str, method: str = "hybrid") -> Dict[str, Any]:
        """
        Trace entire pipeline for a single query
        
        Args:
            raw_query: User's raw query string
            method: "hybrid", "bm25", or "dense"
            
        Returns:
            Dictionary with complete trace data
        """
        
        print("\n" + "=" * 100)
        print("PIPELINE TRACER: Full Query Processing")
        print("=" * 100)
        
        total_start = time.time()
        self.traces = {}
        
        # ========================================
        # STAGE 1: QUERY ENCODING
        # ========================================
        print("\n[1/7] QUERY ENCODING")
        print("-" * 100)
        
        stage_start = time.time()
        try:
            clean_query, dense_vec, sparse_vec = self.query_encoder.encode(raw_query)
            
            self._log_stage("1_query_encoding", {
                "raw_query": raw_query,
                "clean_query": clean_query,
                "dense_vector": self._format_vector(dense_vec),
                "sparse_vector": self._format_sparse_vector(sparse_vec)
            })
            
            print(f"✓ Raw: {raw_query}")
            print(f"✓ Clean: {clean_query}")
            print(f"✓ Dense shape: {len(dense_vec)}")
            print(f"✓ Sparse terms: {len(sparse_vec[0])}")
            print(f"⏱️  Time: {time.time() - stage_start:.4f}s")
            
        except Exception as e:
            logger.error(f"Error in query encoding: {e}")
            return {"error": str(e)}
        
        # ========================================
        # STAGE 2: RETRIEVAL (Based on method)
        # ========================================
        print("\n[2/7] RETRIEVAL STAGE")
        print("-" * 100)
        
        stage_start = time.time()
        
        if method == "hybrid":
            print("Method: HYBRID (Dense + BM25)")
            try:
                dense_results, sparse_results = hybrid_retrieval(dense_vec, sparse_vec, dense_k=100, sparse_k=100)
                
                self._log_stage("2_retrieval_hybrid", {
                    "method": "hybrid",
                    "dense_results": {
                        "count": dense_results["count"],
                        "top_3_scores": dense_results["scores"][:3]
                    },
                    "sparse_results": {
                        "count": sparse_results["count"],
                        "top_3_scores": sparse_results["scores"][:3]
                    }
                })
                
                print(f"✓ Dense chunks: {dense_results['count']}")
                print(f"✓ Sparse chunks: {sparse_results['count']}")
                print(f"  Dense top scores: {[round(s, 4) for s in dense_results['scores'][:3]]}")
                print(f"  Sparse top scores: {[round(s, 4) for s in sparse_results['scores'][:3]]}")
                
                # Prepare for RRF
                all_chunks = []
                for chunk_id, score, payload in zip(
                    dense_results["chunk_ids"],
                    dense_results["scores"],
                    dense_results["payloads"]
                ):
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "dense_rank": len([s for s in dense_results["scores"] if s > score]) + 1,
                        "dense_score": score,
                        "payload": payload
                    })
                
                for chunk_id, score, payload in zip(
                    sparse_results["chunk_ids"],
                    sparse_results["scores"],
                    sparse_results["payloads"]
                ):
                    found = False
                    for chunk in all_chunks:
                        if chunk["chunk_id"] == chunk_id:
                            chunk["sparse_rank"] = len([s for s in sparse_results["scores"] if s > score]) + 1
                            chunk["sparse_score"] = score
                            found = True
                            break
                    if not found:
                        all_chunks.append({
                            "chunk_id": chunk_id,
                            "sparse_rank": len([s for s in sparse_results["scores"] if s > score]) + 1,
                            "sparse_score": score,
                            "payload": payload
                        })
                
            except Exception as e:
                logger.error(f"Error in hybrid retrieval: {e}")
                return {"error": str(e)}
        
        elif method == "bm25":
            print("Method: BM25 ONLY")
            try:
                sparse_results = bm25_retrieval(sparse_vec, k=100)
                
                self._log_stage("2_retrieval_bm25", {
                    "method": "bm25",
                    "count": sparse_results["count"],
                    "top_3_scores": sparse_results["scores"][:3]
                })
                
                print(f"✓ BM25 chunks: {sparse_results['count']}")
                print(f"  Top scores: {[round(s, 4) for s in sparse_results['scores'][:3]]}")
                
                all_chunks = [
                    {
                        "chunk_id": chunk_id,
                        "sparse_rank": idx + 1,
                        "sparse_score": score,
                        "payload": payload
                    }
                    for idx, (chunk_id, score, payload) in enumerate(
                        zip(sparse_results["chunk_ids"], sparse_results["scores"], sparse_results["payloads"])
                    )
                ]
            except Exception as e:
                logger.error(f"Error in BM25 retrieval: {e}")
                return {"error": str(e)}
        
        elif method == "dense":
            print("Method: DENSE ONLY")
            try:
                dense_results = dense_retrieval(dense_vec, k=100)
                
                self._log_stage("2_retrieval_dense", {
                    "method": "dense",
                    "count": dense_results["count"],
                    "top_3_scores": dense_results["scores"][:3]
                })
                
                print(f"✓ Dense chunks: {dense_results['count']}")
                print(f"  Top scores: {[round(s, 4) for s in dense_results['scores'][:3]]}")
                
                all_chunks = [
                    {
                        "chunk_id": chunk_id,
                        "dense_rank": idx + 1,
                        "dense_score": score,
                        "payload": payload
                    }
                    for idx, (chunk_id, score, payload) in enumerate(
                        zip(dense_results["chunk_ids"], dense_results["scores"], dense_results["payloads"])
                    )
                ]
            except Exception as e:
                logger.error(f"Error in Dense retrieval: {e}")
                return {"error": str(e)}
        
        print(f"⏱️  Time: {time.time() - stage_start:.4f}s")
        
        # ========================================
        # STAGE 3: AGGREGATION (Chunks → Movies)
        # ========================================
        print("\n[3/7] AGGREGATION (Chunks → Movies)")
        print("-" * 100)
        
        stage_start = time.time()
        try:
            # Prepare chunks for aggregation
            chunks_for_agg = []
            for chunk in all_chunks:
                chunks_for_agg.append({
                    "chunk_id": chunk["chunk_id"],
                    "rrf_score": chunk.get("rrf_score", chunk.get("dense_score", chunk.get("sparse_score", 0))),
                    "payload": chunk["payload"]
                })
            
            aggregated_movies = self.aggregator_1st.aggregate(chunks_for_agg)
            
            self._log_stage("3_aggregation", {
                "input_chunks": len(chunks_for_agg),
                "output_movies": len(aggregated_movies),
                "top_movies": self._format_movies(aggregated_movies, top_k=5)
            })
            
            print(f"✓ Input: {len(chunks_for_agg)} chunks")
            print(f"✓ Output: {len(aggregated_movies)} movies")
            for i, movie in enumerate(aggregated_movies[:3]):
                print(f"  #{i+1} {movie['title']} (ID: {movie['movie_id']}) - Score: {movie.get('max_score', 0):.4f}")
            
            print(f"⏱️  Time: {time.time() - stage_start:.4f}s")
            
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return {"error": str(e)}
        
        # ========================================
        # STAGE 4: FINAL SCORING
        # ========================================
        print("\n[4/7] FINAL SCORING")
        print("-" * 100)
        
        stage_start = time.time()
        try:
            # Add ce_score for final scoring (using max_score as proxy)
            for movie in aggregated_movies:
                movie["ce_score"] = movie.get("max_score", 0)
            
            final_movies = self.final_scorer.score_and_filter(aggregated_movies, top_n=10)
            
            self._log_stage("4_final_scoring", {
                "input_movies": len(aggregated_movies),
                "output_movies": len(final_movies),
                "top_movies": self._format_movies(final_movies, top_k=10)
            })
            
            print(f"✓ Input: {len(aggregated_movies)} movies")
            print(f"✓ Output: {len(final_movies)} movies (Top-10)")
            for i, movie in enumerate(final_movies[:5]):
                print(f"  #{i+1} {movie['title']} - Final Score: {movie.get('final_score', 0):.4f}")
            
            print(f"⏱️  Time: {time.time() - stage_start:.4f}s")
            
        except Exception as e:
            logger.error(f"Error in final scoring: {e}")
            return {"error": str(e)}
        
        # ========================================
        # STAGE 5: GENERATE REPORT
        # ========================================
        print("\n[5/7] GENERATING REPORT")
        print("-" * 100)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = os.path.join(self.output_dir, f"trace_{timestamp}_query.json")
        csv_filename = os.path.join(self.output_dir, f"trace_{timestamp}_results.csv")
        
        # Save JSON trace
        trace_data = {
            "query": raw_query,
            "method": method,
            "total_time": time.time() - total_start,
            "stages": self.traces,
            "final_results": [
                {
                    "rank": i + 1,
                    "movie_id": m["movie_id"],
                    "title": m["title"],
                    "final_score": round(m.get("final_score", 0), 4),
                    "genres": m.get("genres", ""),
                    "release_date": m.get("release_date", "")
                }
                for i, m in enumerate(final_movies)
            ]
        }
        
        with open(trace_filename, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON trace saved: {trace_filename}")
        
        # Save CSV results
        df_results = pd.DataFrame([
            {
                "rank": i + 1,
                "movie_id": m["movie_id"],
                "title": m["title"],
                "final_score": round(m.get("final_score", 0), 4),
                "genres": m.get("genres", ""),
                "release_date": m.get("release_date", ""),
                "vote_average": m.get("vote_average", 0)
            }
            for i, m in enumerate(final_movies)
        ])
        
        df_results.to_csv(csv_filename, index=False, encoding="utf-8")
        print(f"✓ CSV results saved: {csv_filename}")
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        print("\n" + "=" * 100)
        print("TRACE SUMMARY")
        print("=" * 100)
        
        total_time = time.time() - total_start
        
        print(f"\n📊 Statistics:")
        print(f"  Total Processing Time: {total_time:.4f}s")
        print(f"  Retrieval Time: {self.traces.get('2_retrieval_hybrid' if method == 'hybrid' else f'2_retrieval_{method}', {}).get('data', {})}")
        print(f"  Final Results: Top {len(final_movies)} movies")
        
        print(f"\n🎬 Top 3 Results:")
        for i, movie in enumerate(final_movies[:3]):
            print(f"  #{i+1} [{movie.get('final_score', 0):.4f}] {movie['title']} ({movie.get('release_date', 'N/A')})")
        
        print(f"\n💾 Output Files:")
        print(f"  - {trace_filename}")
        print(f"  - {csv_filename}")
        
        print("\n" + "=" * 100)
        
        return trace_data


def demo_trace(query: str = "action movie about rescue") -> None:
    """Demo tracer with example query"""
    
    tracer = PipelineTracer(output_dir="../evaluation/trace_logs")
    
    # Trace with hybrid method
    print("\n\n" + "=" * 100)
    print("DEMO 1: HYBRID METHOD")
    print("=" * 100)
    trace_hybrid = tracer.trace_query(query, method="hybrid")
    
    # Trace BM25
    print("\n\n" + "=" * 100)
    print("DEMO 2: BM25 METHOD")
    print("=" * 100)
    trace_bm25 = tracer.trace_query(query, method="bm25")
    
    # Trace Dense
    print("\n\n" + "=" * 100)
    print("DEMO 3: DENSE METHOD")
    print("=" * 100)
    trace_dense = tracer.trace_query(query, method="dense")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Tracer - Track query processing")
    parser.add_argument(
        "--query",
        type=str,
        default="action movie about rescue",
        help="Query to trace"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hybrid", "bm25", "dense"],
        default="hybrid",
        help="Retrieval method to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../evaluation/trace_logs",
        help="Output directory for trace logs"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with all 3 methods"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_trace(args.query)
    else:
        tracer = PipelineTracer(output_dir=args.output_dir)
        tracer.trace_query(args.query, method=args.method)
