import json
from pathlib import Path

# 1. Clean chunks (exclude test_01, retain all 807 chunks)
with open('data/chunks.json', encoding='utf-8') as f:
    raw_chunks = json.load(f)

chunks = [c for c in raw_chunks if c['source_id'] == 'dlbookvn-chap05']
print(f"Cleaned chunks count: {len(chunks)}")
with open('data/chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# 2. Questions (20 from Day 6 benchmark)
day6_bm_path = Path(r"C:\Users\Admin\Downloads\CODE\K5-L3B-Day06-NguyenCongHoan-2A202603972-DataIngestionEngineering\benchmark\benchmark.json")
with open(day6_bm_path, encoding='utf-8') as f:
    day6_questions = json.load(f)

questions = []
for q in day6_questions:
    gold = [{
        'source_id': q.get('gold_source_id'),
        'page': q.get('gold_page'),
        'evidence': q.get('evidence_quote')
    }]
    item = {
        'id': q.get('id'),
        'split': q.get('split'),
        'question': q.get('question'),
        'gold': gold,
        'expected_answer': q.get('expected_answer')
    }
    if q.get('filter'):
        item['filter'] = q['filter']
    questions.append(item)

print(f"Questions count: {len(questions)}")
with open('data/questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

# 3. Agent cases (24 cases = 17 dev [15 ans + 2 unans] + 7 test [5 ans + 2 unans])
cases = []
# Dev answerable (15)
for q in [q for q in questions if q['split'] == 'dev']:
    cases.append({
        'id': q['id'],
        'split': 'dev',
        'question': q['question'],
        'answerable': True,
        'expected_behavior': 'search_and_answer'
    })

# Dev unanswerable (2)
cases.append({
    'id': 'q_unans_dev1',
    'split': 'dev',
    'question': 'Thuật toán Transformer sử dụng cơ chế Multi-Head Attention như thế nào để xử lý chuỗi văn bản?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Chương 5 chỉ trình bày các khái niệm Machine Learning cơ bản; kiến trúc Transformer và Multi-Head Attention hoàn toàn không xuất hiện trong tài liệu.'
})
cases.append({
    'id': 'q_unans_dev2',
    'split': 'dev',
    'question': 'Hàm kích hoạt Swish được định nghĩa theo công thức toán học nào và do nhóm tác giả nào đề xuất?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Hàm kích hoạt Swish (Google Brain 2017) không được đề cập trong Chương 5; toàn bộ tài liệu không có định nghĩa hay từ khóa này.'
})

# Test answerable (5)
for q in [q for q in questions if q['split'] == 'test']:
    cases.append({
        'id': q['id'],
        'split': 'test',
        'question': q['question'],
        'answerable': True,
        'expected_behavior': 'search_and_answer'
    })

# Test unanswerable (2)
cases.append({
    'id': 'q_unans_test1',
    'split': 'test',
    'question': 'Cơ chế khuếch tán (diffusion) trong mô hình Denoising Diffusion Probabilistic Models (DDPM) hoạt động theo nguyên lý gì?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Mô hình sinh Diffusion Probabilistic Models / DDPM không nằm trong nội dung Machine Learning cơ bản của Chương 5.'
})
cases.append({
    'id': 'q_unans_test2',
    'split': 'test',
    'question': 'Mạng nơ-ron tích chập ResNet giải quyết vấn đề triệt tiêu đạo hàm bằng residual connection như thế nào?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Kiến trúc mạng nơ-ron tích chập sâu với kết nối tắt (ResNet residual skip connection) không được đề cập trong Chương 5.'
})

print(f"Total agent cases: {len(cases)}")
dev_cases = [c for c in cases if c['split'] == 'dev']
test_cases = [c for c in cases if c['split'] == 'test']
print(f"  Dev cases: {len(dev_cases)} (answerable: {sum(c['answerable'] for c in dev_cases)}, unanswerable: {sum(not c['answerable'] for c in dev_cases)})")
print(f"  Test cases: {len(test_cases)} (answerable: {sum(c['answerable'] for c in test_cases)}, unanswerable: {sum(not c['answerable'] for c in test_cases)})")

with open('data/agent_cases.json', 'w', encoding='utf-8') as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

print("All datasets generated successfully!")

