"""Unit tests for Day 7 solution components: data integrity and retrievers."""
import json
import unittest
from pathlib import Path
from src.retrievers import BM25Retriever, HybridRetriever, tokenize

ROOT = Path(__file__).resolve().parents[1]


class Day7SolutionTests(unittest.TestCase):
    def setUp(self):
        with open(ROOT / 'data/chunks.json', encoding='utf-8') as f:
            self.chunks = json.load(f)
        with open(ROOT / 'data/questions.json', encoding='utf-8') as f:
            self.questions = json.load(f)
        with open(ROOT / 'data/agent_cases.json', encoding='utf-8') as f:
            self.cases = json.load(f)

    def test_corpus_and_questions_contract(self):
        # 807 chunks from Chapter 5
        self.assertEqual(len(self.chunks), 807)
        chunk_ids = {c['id'] for c in self.chunks}
        self.assertEqual(len(chunk_ids), 807)

        # 20 questions (15 dev, 5 test)
        self.assertEqual(len(self.questions), 20)
        dev_q = [q for q in self.questions if q['split'] == 'dev']
        test_q = [q for q in self.questions if q['split'] == 'test']
        self.assertEqual(len(dev_q), 15)
        self.assertEqual(len(test_q), 5)

        # 24 agent cases (17 dev, 7 test)
        self.assertEqual(len(self.cases), 24)
        dev_cases = [c for c in self.cases if c['split'] == 'dev']
        test_cases = [c for c in self.cases if c['split'] == 'test']
        self.assertEqual(len(dev_cases), 17)
        self.assertEqual(len(test_cases), 7)
        self.assertEqual(sum(not c['answerable'] for c in dev_cases), 2)
        self.assertEqual(sum(not c['answerable'] for c in test_cases), 2)

    def test_all_gold_evidence_resolves(self):
        for q in self.questions:
            for g in q['gold']:
                matched = any(
                    c['source_id'] == g['source_id'] and
                    c['page'] == g['page'] and
                    g['evidence'] in c['text']
                    for c in self.chunks
                )
                self.assertTrue(matched, f"Gold evidence failed to resolve for question {q['id']}")

    def test_bm25_retriever(self):
        bm25 = BM25Retriever(self.chunks[:20])
        results = bm25.search("học máy", top_k=3)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertIn('id', r)
            self.assertIn('text', r)
            self.assertIn('score', r)
            self.assertEqual(r['strategy'], 'lexical_bm25')


if __name__ == '__main__':
    unittest.main()

