# Day 7 — Agentic RAG Implementation

Hệ thống Agentic RAG & Retrieval Context Engineering cho Chương 5: Machine Learning cơ bản (Deep Learning Book).

## 1. Cấu trúc mã nguồn

- `src/retrievers.py`: Hiện thực 4 phương pháp retrieval:
  - `DenseRetriever`: Truy vấn vector cosine trên pgvector (PostgreSQL 16) bằng embedding `qwen/qwen3-embedding-8b` (4096 chiều).
  - `BM25Retriever`: Truy vấn từ khóa lexical BM25 với tách từ tiếng Việt chuẩn hóa NFC.
  - `HybridRetriever`: Kết hợp Dense và BM25 thông qua thuật toán Reciprocal Rank Fusion (RRF, $k=60$).
  - `RerankedHybridRetriever` (Cấu hình tối ưu được chọn): Mở rộng candidate pool ($K=15$), kết hợp điểm RRF với tỷ lệ từ khóa trùng khớp (term-overlap ratio) và exact phrase matching, trả về top-k=5 với context budget 1200 ký tự.
- `src/agent.py`: Agent điều phối truy xuất sử dụng mô hình thật `google/gemini-2.5-flash` qua OpenRouter API. Hỗ trợ multi-turn tool calling, trích dẫn chuẩn `[chunk-id]` và cơ chế từ chối (abstain) chính xác khi thiếu bằng chứng.
- `src/evaluate_retrieval.py`: Đo đạc Hit@1, Evidence Hit@3, Context Evidence Rate, latency (ms), chi phí ($).
- `src/run_retrieval_benchmarks.py`: Chạy benchmark toàn bộ 5 cấu hình retrieval trên tập Dev.
- `src/run_test_retrieval.py`: Chạy kiểm thử retrieval độc lập trên tập Test cho Baseline và Cấu hình đã chọn.
- `src/run_agent_benchmarks.py` & `src/run_baseline_agent.py`: Chạy đánh giá Agent thật trên 24 ca benchmark (Dev & Test), thu thập trace, đánh giá grounding và tính toán metrics qua `starter/agent_metrics.py`.

## 2. Dữ liệu Benchmark

- `data/chunks.json`: 807 chunk từ Day 6 (Chương 5 sách Deep Learning tiếng Việt, 73 trang).
- `data/questions.json`: 20 câu hỏi có đáp án vàng (15 Dev, 5 Test).
- `data/agent_cases.json`: 24 ca benchmark (17 Dev [15 ans + 2 unans], 7 Test [5 ans + 2 unans]).
- `data/query_embeddings_cache.json`: Cache vector embedding 4096 chiều cho các câu hỏi và truy vấn của agent.

## 3. Hướng dẫn chạy

```bash
# 1. Chạy toàn bộ Unit Tests
python -m unittest discover -s tests -v

# 2. Chạy Benchmark Retrieval trên Dev
python -m src.run_retrieval_benchmarks

# 3. Chạy Benchmark Retrieval trên Test
python -m src.run_test_retrieval

# 4. Chạy Benchmark Agent đầy đủ (Chosen & Baseline trên Dev và Test)
python -m src.run_agent_benchmarks
python -m src.run_baseline_agent
```

## 4. Báo cáo chi tiết

Báo cáo chi tiết xem tại [`report/individual/2A202603972_NguyenCongHoan.md`](../report/individual/2A202603972_NguyenCongHoan.md).
