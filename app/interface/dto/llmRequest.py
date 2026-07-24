from pydantic import BaseModel
from typing import Optional


class SessionState(BaseModel):
    """현재 세션 상태 정보 (FSM/주문/카트 담당자 작업 완료 후 필드 확정 예정)"""
    se_id: str
    fsm_state: Optional[str] = None        # FSM 담당자 작업 예정
    order_type: Optional[str] = None       # 주문 담당자 작업 예정
    order_item: Optional[dict] = None      # 주문 담당자 작업 예정
    cart: Optional[list] = None            # 카트 담당자 작업 예정
    allergies: Optional[list[str]] = None

class LLMRequest(BaseModel):
    """LLM Service 입력 DTO"""
    session: SessionState     # 현재 세션 정보
    query: str                # 사용자 발화
    context: Optional[str] = ""  # 백엔드가 보낼 때는 비워두고, 라우터 내부에서 RAG 결과로 채워 넣음



