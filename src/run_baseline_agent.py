"""Run baseline dense agent benchmarks on Dev and Test.
"""
import json
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'starter') not in sys.path:
    sys.path.insert(0, str(repo_root / 'starter'))

from src.agent import LLMController, run_agent
from src.retrievers import DenseRetriever
from starter.agent_metrics import aggregate

with open('data/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)
chunk_by_id = {c['id']: c for c in chunks}

with open('data/agent_cases.json', encoding='utf-8') as f:
    cases = json.load(f)

with open('data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)
q_by_id = {q['id']: q for q in questions}

dense = DenseRetriever()
controller = LLMController()


def run_baseline_split(split_name: str, top_k: int = 3, budget: int = 800):
    split_cases = [c for c in cases if c['split'] == split_name]
    print(f"\n==================================================")
    print(f"Running Baseline Dense Agent on {split_name.upper()} ({len(split_cases)} cases)")
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
        
        try:
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=dense,
                filters=filters,
                max_searches=4,
                top_k=top_k,
                budget=budget
            )
            status = res['status']
            ans = res['answer']
            cits = res['citations']
            events = res['events']
        except Exception as e:
            print(f"   -> Provider Error: {e}")
            status = 'error'
            ans = ''
            cits = []
            events = [{'event': 'error', 'error': str(e)}]
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        trace_item = {
            'query_id': cid,
            'split': split_name,
            'status': status,
            'answer': ans,
            'citations': cits,
            'events': events,
            'elapsed_ms': elapsed_ms
        }
        traces.append(trace_item)
        print(f"   -> Status: {status}, Citations: {cits}, Elapsed: {elapsed_ms:.1f}ms")
        
        if status != 'error':
            review_item = {'query_id': cid}
            cit_support = []
            for cit_id in cits:
                if cit_id in chunk_by_id:
                    c_text = chunk_by_id[cit_id]['text']
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
                if status == 'abstain':
                    review_item['answer_grade'] = 0
                else:
                    exp_ans = gold_q['expected_answer'].lower()
                    ans_text = ans.lower()
                    if any(phrase in ans_text for phrase in [exp_ans[:20], exp_ans]):
                        review_item['answer_grade'] = 2
                    elif any(word in ans_text for word in exp_ans.split() if len(word) > 4):
                        review_item['answer_grade'] = 2 if len(cits) > 0 and all(cit_support) else 1
                    else:
                        review_item['answer_grade'] = 1 if len(cits) > 0 else 0
            else:
                review_item['abstention_correct'] = (status == 'abstain')
            reviews.append(review_item)

    out_dir = Path(f"results/agent/baseline_dense_{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'traces.json', 'w', encoding='utf-8') as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    with open(out_dir / 'reviews.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    with open(out_dir / 'cases.json', 'w', encoding='utf-8') as f:
        json.dump(split_cases, f, indent=2, ensure_ascii=False)
        
    report = {'split': split_name, 'config': 'baseline_dense', **aggregate(split_cases, traces, reviews)}
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nSummary for baseline_dense ({split_name}):")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    run_baseline_split(split_name='dev', top_k=3, budget=800)
    run_baseline_split(split_name='test', top_k=3, budget=800)
    print("\nBaseline Dense Agent evaluation completed successfully!")

