"""Run retrieval evaluation on Test set for Baseline and Selected Configuration.
"""
import json
from pathlib import Path
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedHybridRetriever
from src.evaluate_retrieval import evaluate_method

with open('data/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

with open('data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)

dense = DenseRetriever()
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
reranked = RerankedHybridRetriever(hybrid)

out_dir = Path('results/retrieval')

# 1. Baseline on Test (Dense, top_k=3, budget=800)
print("\n--- Evaluating Baseline: Dense on Test ---")
eval_dense_test = evaluate_method(dense, questions, method_name='baseline_dense_test', split='test',
                                  top_k=3, budget=800, cost_per_query=0.00001)
print(json.dumps(eval_dense_test['summary'], indent=2, ensure_ascii=False))

# 2. Selected Configuration on Test (Reranked Hybrid, top_k=5, budget=1200)
print("\n--- Evaluating Selected Configuration: Reranked Hybrid on Test ---")
eval_selected_test = evaluate_method(reranked, questions, method_name='selected_reranked_hybrid_test', split='test',
                                     top_k=5, budget=1200, cost_per_query=0.00001)
print(json.dumps(eval_selected_test['summary'], indent=2, ensure_ascii=False))

for name, res in [('baseline_dense_test', eval_dense_test), ('selected_reranked_hybrid_test', eval_selected_test)]:
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(res['summary'], f, indent=2, ensure_ascii=False)
    with open(exp_dir / 'traces.json', 'w', encoding='utf-8') as f:
        json.dump(res['traces'], f, indent=2, ensure_ascii=False)

print("\nTest retrieval evaluation finished successfully!")

