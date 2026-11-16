"""Real Agent implementation using OpenRouter LLM controller and retrieval adapters.
Compatible with starter/agentic.py and starter/agent_metrics.py.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / 'starter') not in sys.path:
    sys.path.insert(0, str(repo_root / 'starter'))

from starter.baseline import pack_context
from starter.agentic import run_agent

load_dotenv('src/.env')


class LLMController:
    """Controller that calls a real LLM (google/gemini-2.5-flash) via OpenRouter to decide agent actions."""
    def __init__(self, api_key: Optional[str] = None, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.session = requests.Session()
        self.total_tokens = 0
        self.total_calls = 0

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call OpenRouter LLM to return either search or answer action."""
        system_prompt = (
            "Bạn là một AI Agent nghiên cứu và truy xuất thông tin (Agentic RAG) chuyên nghiệp.\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi dựa HOÀN TOÀN vào các đoạn văn bản (context) được cung cấp.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Nếu chưa có context nào được cung cấp, bạn PHẢI chọn action 'search' với query ngắn gọn, tập trung vào từ khóa của câu hỏi.\n"
            "2. Nếu context hiện tại đã có ĐỦ bằng chứng chính xác để trả lời, bạn chọn action 'answer'.\n"
            "   - Trong câu trả lời, bạn BẮT BUỘC phải trích dẫn ID của chunk chứa bằng chứng dưới dạng [chunk-id]. Ví dụ: [dlbookvn-chap05_p3_fa000].\n"
            "   - Danh sách 'citations' phải chứa đúng các ID chunk đó.\n"
            "3. Nếu context đã tìm kiếm nhưng KHÔNG có bằng chứng, hoặc tài liệu không đề cập đến nội dung câu hỏi, bạn PHẢI chọn action 'answer' với:\n"
            "   'answer': 'Không đủ bằng chứng để trả lời.' và 'citations': [].\n"
            "4. ĐỊNH DẠNG ĐẦU RA: Bạn CHỈ được trả về duy nhất một JSON object hợp lệ (không markdown block, không giải thích ngoài JSON):\n"
            '   Search: {"action": "search", "query": "..."}\n'
            '   Answer: {"action": "answer", "answer": "... [chunk-id]", "citations": ["chunk-id"]}\n'
            '   Abstain: {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
        )
        
        user_msg = f"Câu hỏi: {question}\n\n"
        if not history:
            user_msg += "Lịch sử tìm kiếm: Chưa có. Vui lòng thực hiện tìm kiếm đầu tiên."
        else:
            last = history[-1]
            user_msg += f"Các truy vấn đã tìm: {last.get('queries', [])}\n"
            user_msg += f"Context hiện tại:\n{last.get('context', '')}\n"
            user_msg += "\nDựa trên context trên, hãy quyết định: tìm kiếm tiếp (nếu cần thêm thông tin) hoặc trả lời có trích dẫn [chunk-id] hoặc từ chối nếu không có bằng chứng."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        for attempt in range(3):
            try:
                resp = self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(5.0, 25.0)
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.total_calls += 1
                    usage = data.get('usage', {})
                    self.total_tokens += usage.get('total_tokens', 0)
                    content = data['choices'][0]['message']['content'].strip()
                    
                    if content.startswith('```'):
                        content = re.sub(r'^```(?:json)?\s*', '', content)
                        content = re.sub(r'\s*```$', '', content)
                    parsed = json.loads(content)
                    
                    action = parsed.get('action')
                    if action == 'search':
                        q = parsed.get('query', '').strip() or question
                        return {'action': 'search', 'query': q}
                    elif action == 'answer':
                        ans = parsed.get('answer', '')
                        cits = parsed.get('citations', [])
                        if not isinstance(cits, list):
                            cits = [cits] if cits else []
                        return {'action': 'answer', 'answer': ans, 'citations': cits}
                    else:
                        return {'action': 'search', 'query': question}
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
                
        return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}
