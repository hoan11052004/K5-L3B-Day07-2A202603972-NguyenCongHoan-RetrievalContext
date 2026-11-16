"""Evaluate retrieval methods on dev and test sets.
Calculates Hit@1, Evidence Hit@3, Context Evidence Rate, avg context length, latency, and cost.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedHybridRetriever


def hit(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    return any(
        chunk['source_id'] == g['source_id'] and
        chunk['page'] == g['page'] and
        g['evidence'] in chunk['text']
        for g in gold
    )


def pack_context(results: List[Dict[str, Any]], budget: int = 800) -> tuple[str, List[str]]:
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate_method(retriever, questions: List[Dict[str, Any]], *,
                    method_name: str, split: str, top_k: int = 3, budget: int = 800,
                    cost_per_query: float = 0.0) -> Dict[str, Any]:
    traces = []
    selected = [q for q in questions if q.get('split') == split]
    
    start_total = time.perf_counter()
    for q in selected:
        q_start = time.perf_counter()
        
        # Support search with filters if question specifies filter
        filters = q.get('filter')
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - q_start) * 1000
        
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        h1 = bool(marked and marked[0]['hit'])
        h3 = any(c['hit'] for c in marked[:3])
        c_hit = any(c['hit'] and c['id'] in ids for c in marked)
        
        traces.append({
            'query_id': q['id'],
            'split': split,
            'question': q['question'],
            'strategy': method_name,
            'top_k': top_k,
            'budget_chars': budget,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': h1,
            'hit3': h3,
            'context_hit': c_hit
        })
        
    n = len(traces)
    total_cost = cost_per_query * n
    summary = {
        'strategy': method_name,
        'split': split,
        'n': n,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_rate': sum(t['hit1'] for t in traces) / (n or 1),
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_rate': sum(t['hit3'] for t in traces) / (n or 1),
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_evidence_rate': sum(t['context_hit'] for t in traces) / (n or 1),
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0,
        'mean_ms': statistics.mean(t['elapsed_ms'] for t in traces) if traces else 0,
        'service_cost_usd': total_cost
    }
    return {'summary': summary, 'traces': traces}

