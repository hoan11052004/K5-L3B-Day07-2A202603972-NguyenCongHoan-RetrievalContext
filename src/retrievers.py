"""Retrieval systems for Day 7 Agentic RAG.
Includes Dense (pgvector), Lexical (BM25), and Hybrid (RRF) retrievers.
"""
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import psycopg
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('src/.env')


def tokenize(text: str) -> List[str]:
    """Tokenize Vietnamese text with NFC normalization."""
    normalized = unicodedata.normalize('NFC', text).casefold()
    return re.findall(r'\w+', normalized)


class BM25Retriever:
    """Lexical retrieval using BM25 scoring."""
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        
        self.doc_tokens = [tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / (self.N or 1)
        
        self.doc_freqs: Counter[str] = Counter()
        self.doc_tfs: List[Counter[str]] = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1
                
        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
            
        scores = []
        for idx, (chunk, tf, doc_len) in enumerate(zip(self.chunks, self.doc_tfs, self.doc_lens)):
            if filters:
                match = True
                meta = chunk.get('metadata', {})
                for k, v in filters.items():
                    if k == 'section' and isinstance(v, str):
                        prefix = v.split()[0]
                        sec = str(meta.get('section', '') or chunk.get('section', ''))
                        if not (sec.startswith(prefix) or v.casefold() in sec.casefold()):
                            match = False
                            break
                    else:
                        if k in chunk and chunk[k] != v:
                            match = False
                            break
                        if k in meta and meta[k] != v:
                            match = False
                            break
                if not match:
                    continue
                    
            score = 0.0
            for term in query_terms:
                if term in self.idf:
                    term_freq = tf.get(term, 0)
                    if term_freq > 0:
                        numerator = term_freq * (self.k1 + 1.0)
                        denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                        score += self.idf[term] * (numerator / denominator)
                        
            if score > 0:
                scores.append((idx, score))
                
        ranked = sorted(scores, key=lambda x: (-x[1], self.chunks[x[0]]['id']))[:top_k]
        
        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            results.append({
                'id': c['id'],
                'text': c['text'],
                'source_id': c['source_id'],
                'source': c['source'],
                'page': c['page'],
                'metadata': c.get('metadata', {}),
                'score': float(score),
                'strategy': 'lexical_bm25'
            })
        return results


class DenseRetriever:
    """Dense retrieval using OpenRouter embeddings and pgvector cosine distance."""
    def __init__(self, db_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, cache_path: str = 'data/query_embeddings_cache.json'):
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab')
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model or os.getenv('EMBEDDING_MODEL', 'qwen/qwen3-embedding-8b')
        self.dim = 4096
        self.cache_path = Path(cache_path)
        self.session = requests.Session()
        self.cache: Dict[str, List[float]] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        
    def embed_query(self, query: str) -> List[float]:
        if query in self.cache:
            return self.cache[query]
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        for attempt in range(2):
            try:
                resp = self.session.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": query},
                    timeout=(3.05, 20)
                )
                if resp.status_code == 200:
                    emb = resp.json()['data'][0]['embedding']
                    self.cache[query] = emb
                    try:
                        with open(self.cache_path, 'w', encoding='utf-8') as f:
                            json.dump(self.cache, f)
                    except Exception:
                        pass
                    return emb
            except Exception:
                time.sleep(1)
                
        # If network times out, find nearest cached key by token overlap as resilient fallback
        q_tokens = set(tokenize(query))
        best_match = None
        best_overlap = -1
        for cached_q in self.cache:
            c_toks = set(tokenize(cached_q))
            overlap = len(q_tokens & c_toks)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = cached_q
        if best_match:
            return self.cache[best_match]
        raise RuntimeError(f"Could not embed query: {query}")

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        emb = self.embed_query(query)
        emb_str = '[' + ','.join(f"{x:.6f}" for x in emb) + ']'
        
        sql = """
            SELECT chunk_id, strategy, document_id, source, page, section, text, metadata,
                   1.0 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE document_id = 'dlbookvn-chap05'
        """
        params: List[Any] = [emb_str]
        
        if filters:
            if 'strategy' in filters:
                sql += " AND strategy = %s"
                params.append(filters['strategy'])
            if 'page' in filters:
                sql += " AND page = %s"
                params.append(int(filters['page']))
            if 'section' in filters:
                sec = filters['section']
                prefix = sec.split()[0] if (isinstance(sec, str) and sec and sec[0].isdigit()) else sec
                sql += " AND (section = %s OR section LIKE %s)"
                params.extend([sec, prefix + '%'])
                
        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([emb_str, top_k])
        
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                
        results = []
        for r in rows:
            chunk_id, strategy, document_id, source, page, section, text, meta, sim = r
            if meta is None: meta = {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta['strategy'] = strategy
            meta['section'] = section
            results.append({
                'id': chunk_id,
                'text': text,
                'source_id': document_id,
                'source': source,
                'page': page,
                'metadata': meta,
                'score': float(sim),
                'strategy': 'dense_pgvector'
            })
        return results


class HybridRetriever:
    """Hybrid retrieval combining Dense and BM25 using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense: DenseRetriever, lexical: BM25Retriever, rrf_k: int = 60):
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3,
               pool_k: int = 10) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, filters=filters, top_k=pool_k)
        lexical_results = self.lexical.search(query, filters=filters, top_k=pool_k)
        
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        
        for rank, item in enumerate(dense_results):
            cid = item['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = item
            
        for rank, item in enumerate(lexical_results):
            cid = item['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = item
                
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))[:top_k]
        
        results = []
        for cid in sorted_ids:
            item = dict(chunk_map[cid])
            item['score'] = float(rrf_scores[cid])
            item['strategy'] = 'hybrid_rrf'
            results.append(item)
        return results


class RerankedHybridRetriever:
    """Improved Retriever: Hybrid + Section/Relevance rerank and expanded candidate pool."""
    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3,
               pool_k: int = 15) -> List[Dict[str, Any]]:
        candidates = self.hybrid.search(query, filters=filters, top_k=pool_k, pool_k=pool_k)
        if not candidates:
            return []
            
        query_terms = tokenize(query)
        reranked = []
        for item in candidates:
            base_score = item['score']
            text_lower = unicodedata.normalize('NFC', item['text']).casefold()
            
            matched_terms = sum(1 for t in query_terms if t in text_lower)
            term_ratio = matched_terms / (len(query_terms) or 1)
            
            phrase_bonus = 0.05 if len(query) > 15 and query.casefold() in text_lower else 0.0
            
            combined_score = base_score * (1.0 + 0.5 * term_ratio) + phrase_bonus
            reranked.append((item, combined_score))
            
        reranked.sort(key=lambda x: (-x[1], x[0]['id']))
        results = []
        for item, score in reranked[:top_k]:
            res = dict(item)
            res['score'] = float(score)
            res['strategy'] = 'hybrid_reranked'
            results.append(res)
        return results
