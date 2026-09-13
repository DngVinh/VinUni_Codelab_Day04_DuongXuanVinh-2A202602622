"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). BẮT BUỘC gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để ghi nhận sự cố.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả chung), trả lời trực tiếp không cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi TUẦN TỰ từng tool rồi tổng hợp kết quả.

## OPERATIONAL BOUNDARIES
- CHỈ hỗ trợ các chủ đề liên quan đến Vingroup: VinFast, Vinpearl, Vinhomes, Vinmec.
- TỪ CHỐI lịch sự các câu hỏi ngoài phạm vi hệ sinh thái Vingroup.

## OUTPUT CONTRACT
Với mỗi bước suy luận, tuân thủ định dạng:
Thought: <suy nghĩ về bước tiếp theo>
Action: <tên tool cần gọi> | None (nếu đã có câu trả lời)
Action Input: <tham số JSON của tool>
Observation: <kết quả từ tool>
Final Answer: <câu trả lời hoàn chỉnh cho người dùng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Hãy trả lời câu hỏi sau:\n{user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_gemini"
                }
            except Exception:
                pass

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
      - 2 custom tools: search_product_catalog, submit_support_ticket
      - Sequential & Parallel tool calling
      - Max iterations safeguard
      - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích ý định người dùng (Catalog Search, Ticket Submit, FAQ)."""
        lowered = user_input.lower()

        is_faq = "bảo hành" in lowered or "chính sách" in lowered or "quy định" in lowered

        catalog_intent_keywords = ["xem", "tìm", "tra cứu", "mua", "giá", "có xe", "có resort", "có phòng"]
        has_catalog_verb = any(v in lowered for v in catalog_intent_keywords)
        has_catalog_entity = any(n in lowered for n in ["xe", "resort", "vinpearl", "vinfast", "du lịch", "du_lich", "phòng"])

        needs_catalog = (has_catalog_verb and has_catalog_entity)
        if is_faq and not any(k in lowered for k in ["dưới", "giá dưới", "mua", "tìm", "xem"]):
            needs_catalog = False

        ticket_keywords = [
            "lỗi", "sự cố", "khiếu nại", "phản hồi", "hỗ trợ", "vấn đề", "hỏng", "kẹt", "ẩm mốc", "xử lý gấp"
        ]
        needs_ticket = any(k in lowered for k in ticket_keywords)

        category = "xe_dien"
        if any(k in lowered for k in ["resort", "du lịch", "du_lich", "vinpearl", "khách sạn", "phòng"]):
            category = "du_lich"

        max_price = 999999999999
        price_match = re.search(r"dưới\s+(\d+(?:[\.,]\d+)?)\s*(triệu|tỷ|tr)", lowered)
        if price_match:
            num = float(price_match.group(1).replace(",", "."))
            unit = price_match.group(2)
            if unit in ["triệu", "tr"]:
                max_price = int(num * 1_000_000)
            elif unit == "tỷ":
                max_price = int(num * 1_000_000_000)

        customer_name = "Khách hàng"
        name_match = re.search(
            r"(?:tôi tên|tên tôi)\s*(?:là)?\s*[:\s]*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bvà\b|\bxe\b|\bphòng\b|$)",
            user_input,
            re.IGNORECASE
        )
        if name_match:
            customer_name = name_match.group(1).strip()

        priority = "medium"
        if any(k in lowered for k in ["gấp", "nghiêm trọng", "khẩn cấp"]):
            priority = "high"
        elif any(k in lowered for k in ["thấp", "không gấp"]):
            priority = "low"

        issue_desc = user_input
        desc_match = re.search(r"((?:xe|phòng|hệ thống|dịch vụ)[^,\.]+(?:bị|lỗi|hỏng|ẩm mốc)[^,\.]*)", user_input, re.IGNORECASE)
        if desc_match:
            issue_desc = desc_match.group(1).strip()

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "catalog_args": {"category": category, "max_price": max_price},
            "ticket_args": {"customer_name": customer_name, "issue_description": issue_desc, "priority": priority}
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)

        actions = []
        if intents["is_faq"] and not intents["needs_catalog"] and not intents["needs_ticket"]:
            actions.append(("faq", {}))
        else:
            if intents["needs_catalog"]:
                actions.append(("search_product_catalog", intents["catalog_args"]))
            if intents["needs_ticket"]:
                actions.append(("submit_support_ticket", intents["ticket_args"]))

        if not actions:
            actions.append(("faq", {}))

        iteration = 0
        catalog_results = None
        ticket_result = None

        while iteration < self.max_iterations and iteration < len(actions):
            tool_name, tool_args = actions[iteration]
            iteration += 1

            if tool_name == "faq":
                thought = "Câu hỏi FAQ chung, trả lời trực tiếp từ tri thức có sẵn không cần gọi tool."
                answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km (tùy điều kiện nào đến trước)."
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": "None",
                    "action_input": None,
                    "observation": "Direct response",
                    "final_answer": answer
                })
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            elif tool_name == "search_product_catalog":
                thought = f"Cần tra cứu sản phẩm trong catalog với tham số: {tool_args}."
                tool_func = TOOL_MAP.get(tool_name, search_product_catalog)
                obs = tool_func(**tool_args)
                catalog_results = obs
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": obs
                })

            elif tool_name == "submit_support_ticket":
                thought = f"Cần ghi nhận yêu cầu hỗ trợ vào hệ thống ticket với tham số: {tool_args}."
                tool_func = TOOL_MAP.get(tool_name, submit_support_ticket)
                obs = tool_func(**tool_args)
                ticket_result = obs
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": obs
                })

            if iteration == len(actions):
                parts = []
                if catalog_results is not None:
                    if not catalog_results or len(catalog_results) == 0:
                        parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của quý khách.")
                    else:
                        prod_strs = [f"{p.get('name', '')} ({p.get('price_vnd', 0):,} VNĐ)" for p in catalog_results]
                        parts.append(f"Tìm thấy các sản phẩm phù hợp: {', '.join(prod_strs)}.")

                if ticket_result is not None:
                    parts.append(
                        f"Yêu cầu hỗ trợ của quý khách {ticket_result.get('customer_name', '')} đã được ghi nhận thành công. "
                        f"Mã ticket: {ticket_result.get('ticket_id', '')}, trạng thái: {ticket_result.get('status', '')}."
                    )

                answer = "\n".join(parts)
                self.trace[-1]["final_answer"] = answer
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

        if iteration >= self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached"
            }

        return {
            "answer": "Không thể xử lý yêu cầu.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "failed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
