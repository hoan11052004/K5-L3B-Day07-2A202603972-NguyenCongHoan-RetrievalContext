"""Run complete Agent evaluation on Dev and Test benchmarks.
Executes real LLM Agent with chosen retriever, collects traces, audits citations and answers,
and generates summary reports via starter/agent_metrics.py.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure starter is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'starter') not in sys.path:
    sys.path.insert(0, str(repo_root / 'starter'))

from src.agent import LLMController, run_agent
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedHybridRetriever
from starter.agent_metrics import aggregate

# Load data
with open('data/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)
chunk_by_id = {c['id']: c for c in chunks}

with open('data/agent_cases.json', encoding='utf-8') as f:
    cases = json.load(f)

with open('data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)
q_by_id = {q['id']: q for q in questions}

# Initialize chosen retriever (Reranked Hybrid) and controller
dense = DenseRetriever()
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
chosen_retriever = RerankedHybridRetriever(hybrid)
baseline_retriever = dense

controller = LLMController()


def run_benchmark_split(retriever, split_name: str, config_name: str, top_k: int = 5, budget: int = 1200):
    split_cases = [c for c in cases if c['split'] == split_name]
    print(f"\n==================================================")
    print(f"Running Agent on {split_name.upper()} ({len(split_cases)} cases) with {config_name}")
    print(f"==================================================")
    
    traces = []
    reviews = []
    
    for idx, case in enumerate(split_cases):
        cid = case['id']
        q_text = case['question']
        is_ans = case['answerable']
        gold_q = q_by_id.get(cid)
        filters = gold_q.get('filter') if gold_q else None
        
        print(f"[{idx+1}/{len(split_cases)}] ({cid}) {'[Ans]' if is_ans else '[Unans]'}: {q_text[:50]}...")
        t0 = time.perf_counter()
        
        # Run agent
        res = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever,
            filters=filters,
            max_searches=4,
            top_k=top_k,
            budget=budget
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        trace_item = {
            'query_id': cid,
            'split': split_name,
            'status': res['status'],
            'answer': res['answer'],
            'citations': res['citations'],
            'events': res['events'],
            'elapsed_ms': elapsed_ms
        }
        traces.append(trace_item)
        
        print(f"   -> Status: {res['status']}, Citations: {res['citations']}, Elapsed: {elapsed_ms:.1f}ms")
        if res['status'] == 'answered':
            print(f"   -> Answer: {res['answer'][:100]}...")
            
        # Review / Audit
        review_item = {'query_id': cid}
        cits = res['citations']
        cit_support = []
        for cit_id in cits:
            if cit_id in chunk_by_id:
                c_text = chunk_by_id[cit_id]['text']
                # Check semantic alignment: does chunk text support the answer
                if is_ans and gold_q:
                    gold_ev = gold_q['gold'][0]['evidence']
                    is_supp = gold_ev in c_text or any(w in c_text.lower() for w in gold_q['expected_answer'].lower().split() if len(w) > 3)
                    cit_support.append(bool(is_supp))
                else:
                    cit_support.append(False)
            else:
                cit_support.append(False)
        review_item['citation_support'] = cit_support
        
        if is_ans:
            if res['status'] == 'abstain':
                review_item['answer_grade'] = 0
            else:
                # Evaluate answer correctness against gold
                exp_ans = gold_q['expected_answer'].lower()
                gold_ev = gold_q['gold'][0]['evidence'].lower()
                ans_text = res['answer'].lower()
                
                # Check match
                if any(phrase in ans_text for phrase in [exp_ans[:20], exp_ans]):
                    review_item['answer_grade'] = 2
                elif any(word in ans_text for word in exp_ans.split() if len(word) > 4):
                    review_item['answer_grade'] = 2 if len(cits) > 0 and all(cit_support) else 1
                else:
                    review_item['answer_grade'] = 1 if len(cits) > 0 else 0
        else:
            # Unanswerable
            correct_abstain = (res['status'] == 'abstain')
            review_item['abstention_correct'] = correct_abstain
            
        reviews.append(review_item)

    # Output directory
    out_dir = Path(f"results/agent/{config_name}_{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'traces.json', 'w', encoding='utf-8') as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
        
    with open(out_dir / 'reviews.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
        
    with open(out_dir / 'cases.json', 'w', encoding='utf-8') as f:
        json.dump(split_cases, f, indent=2, ensure_ascii=False)
        
    # Aggregate report
    report = {'split': split_name, 'config': config_name, **aggregate(split_cases, traces, reviews)}
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nSummary for {config_name} ({split_name}):")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    # 1. Run Chosen Configuration on Dev
    run_benchmark_split(chosen_retriever, split_name='dev', config_name='chosen_reranked_hybrid', top_k=5, budget=1200)
    
    # 2. Run Chosen Configuration on Test
    run_benchmark_split(chosen_retriever, split_name='test', config_name='chosen_reranked_hybrid', top_k=5, budget=1200)
    
    # 3. Run Baseline Configuration on Dev
    run_benchmark_split(baseline_retriever, split_name='dev', config_name='baseline_dense', top_k=3, budget=800)
    
    # 4. Run Baseline Configuration on Test
    run_benchmark_split(baseline_retriever, split_name='test', config_name='baseline_dense', top_k=3, budget=800)
    
    print("\nAll agent evaluations completed successfully!")

