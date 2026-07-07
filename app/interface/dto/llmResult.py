from pydantic import BaseModel
from typing import List


class FSMEvent(BaseModel):
    """FSM에서 실행할 이벤트. 백엔드 app.fsm.event.FSMEvent 구조와 일치."""
    type: str               # 백엔드 Event Enum값 (예: SELECT_MENU, ADD_CART 등)
    parameters: dict = {}   # 이벤트 실행에 필요한 파라미터


class LLMResult(BaseModel):
    """LLM Service 출력 DTO"""
    result: str                   # LLM 자연어 응답 원문
    events: List[FSMEvent] = []   # FSM 이벤트 목록 (없으면 빈 리스트)
    response: str                 # Android에 전달할 안내 문구
    source: str = "LLM"           # 처리 출처