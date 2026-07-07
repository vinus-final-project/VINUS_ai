from pathlib import Path
from app.interface.dto.llmRequest import LLMRequest
import logging

logger = logging.getLogger(__name__)

# 추천 판단 키워드 목록
RECOMMEND_KEYWORDS = [
    "추천", "추천해", "추천해줘", "추천좀",
    "뭐가 좋아", "뭐가 맛있어", "뭐 먹을까",
    "골라줘", "골라", "어떤 게 좋아", "어떤게 좋아",
    "뭐가 나아", "뭐로 할까", "뭐 시킬까",
    "맛있는 거", "인기 있는 거", "인기있는 거"
]

# 프롬프트 파일 경로
PROMPT_DIR = Path(__file__).parent / "prompt"
SYSTEM_PROMPT_PATH = PROMPT_DIR / "systemPrompt.txt"
RECOMMENDATION_PROMPT_PATH = PROMPT_DIR / "recommendationPrompt.txt"


def _load_prompt_file(path: Path) -> str:
    """프롬프트 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_recommendation_query(query: str) -> bool:
    """추천 요청 여부 판단"""
    return any(keyword in query for keyword in RECOMMEND_KEYWORDS)


def _format_session_info(request: LLMRequest) -> str:
    """Session 정보를 프롬프트용 문자열로 변환"""
    session = request.session
    return f"""
[현재 주문 상태]
- FSM 상태: {session.fsm_state or "INIT"}
- 주문 유형: {session.order_type or "미선택"}
- 현재 주문중인 메뉴: {session.order_item or "없음"}
- 장바구니: {session.cart or "비어있음"}
""".strip()


def build_prompt_promptBuilder(request: LLMRequest) -> list[dict]:
    """
    LLMRequest를 받아 EXAONE용 메시지 리스트 생성
    반환 형식: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    is_recommendation = _is_recommendation_query(request.query)

    # 시스템 프롬프트 선택
    if is_recommendation:
        logger.info(f"추천 프롬프트 사용 | query: {request.query}")
        system_prompt = _load_prompt_file(RECOMMENDATION_PROMPT_PATH)
    else:
        logger.info(f"일반 프롬프트 사용 | query: {request.query}")
        system_prompt = _load_prompt_file(SYSTEM_PROMPT_PATH)

    # 세션 정보 + Context + Query 조합
    user_content = f"""
{_format_session_info(request)}

[관련 메뉴 정보 (Context)]
{request.context if request.context else "관련 정보 없음"}

[사용자 발화]
{request.query}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]