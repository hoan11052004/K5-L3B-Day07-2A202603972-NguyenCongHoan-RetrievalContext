"""Run all retrieval evaluations on Dev set and save traces and summaries.
"""
import json
import os
from pathlib import Path
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedHybridRetriever
from src.evaluate_retrieval import evaluate_method

# Load chunks and questions
with open('data/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

with open('data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)

print(f"Loaded {len(chunks)} chunks and {len(questions)} questions.")

# Initialize retrievers
bm25 = BM25Retriever(chunks)
dense = DenseRetriever()
hybrid = HybridRetriever(dense, bm25)
reranked = RerankedHybridRetriever(hybrid)

out_dir = Path('results/retrieval')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Baseline: Dense Retrieval (pgvector, top_k=3, budget=800)
print("\n--- Evaluating Baseline: Dense Retrieval (top_k=3, budget=800) on dev ---")
eval_dense_dev = evaluate_method(dense, questions, method_name='dense_pgvector', split='dev',
                                 top_k=3, budget=800, cost_per_query=0.00001)
print(json.dumps(eval_dense_dev['summary'], indent=2, ensure_ascii=False))

# 2. Method 2: Lexical Retrieval (BM25, top_k=3, budget=800)
print("\n--- Evaluating Method 2: BM25 (top_k=3, budget=800) on dev ---")
eval_bm25_dev = evaluate_method(bm25, questions, method_name='lexical_bm25', split='dev',
                                top_k=3, budget=800, cost_per_query=0.0)
print(json.dumps(eval_bm25_dev['summary'], indent=2, ensure_ascii=False))

# 3. Method 3: Hybrid Retrieval (Dense + BM25 via RRF, top_k=3, budget=800)
print("\n--- Evaluating Method 3: Hybrid RRF (top_k=3, budget=800) on dev ---")
eval_hybrid_dev = evaluate_method(hybrid, questions, method_name='hybrid_rrf', split='dev',
                                  top_k=3, budget=800, cost_per_query=0.00001)
print(json.dumps(eval_hybrid_dev['summary'], indent=2, ensure_ascii=False))

# 4. Improvement 1: Hybrid RRF + Budget 1200 (top_k=3, budget=1200)
print("\n--- Evaluating Improvement 1: Hybrid RRF (top_k=3, budget=1200) on dev ---")
eval_imp1_dev = evaluate_method(hybrid, questions, method_name='hybrid_rrf_b1200', split='dev',
                                top_k=3, budget=1200, cost_per_query=0.00001)
print(json.dumps(eval_imp1_dev['summary'], indent=2, ensure_ascii=False))

# 5. Improvement 2: Reranked Hybrid Retrieval (top_k=5, budget=1200)
print("\n--- Evaluating Improvement 2: Reranked Hybrid (top_k=5, budget=1200) on dev ---")
eval_imp2_dev = evaluate_method(reranked, questions, method_name='hybrid_reranked_k5_b1200', split='dev',
                                top_k=5, budget=1200, cost_per_query=0.00001)
print(json.dumps(eval_imp2_dev['summary'], indent=2, ensure_ascii=False))

# Save all results
experiments = {
    'baseline_dense_dev': eval_dense_dev,
    'method2_bm25_dev': eval_bm25_dev,
    'method3_hybrid_dev': eval_hybrid_dev,
    'improvement1_hybrid_b1200_dev': eval_imp1_dev,
    'improvement2_reranked_k5_b1200_dev': eval_imp2_dev,
}

for name, res in experiments.items():
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(res['summary'], f, indent=2, ensure_ascii=False)
    with open(exp_dir / 'traces.json', 'w', encoding='utf-8') as f:
        json.dump(res['traces'], f, indent=2, ensure_ascii=False)

print("\nAll dev retrieval benchmarks completed and saved successfully!")

