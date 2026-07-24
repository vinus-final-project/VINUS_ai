import asyncio
import traceback
from fastapi import APIRouter, HTTPException
from app.interface.dto.llmRequest import LLMRequest
from app.interface.dto.llmResult import LLMResult
from app.rag.ragService import get_rag_ragService
from app.llm.llmService import LLMService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/v1/llm", response_model=LLMResult)
async def generate_llm_result(request: LLMRequest) -> LLMResult:
    """백엔드 AI 클라이언트 요청 수신 → LLMResult 반환"""

    logger.info(f"LLM 요청 수신 | session_id: {request.session.se_id} | query: {request.query}")

    try:
        # 1. RAG 검색 → context 생성 (동기 함수라 스레드풀에서 실행)
        rag_service = get_rag_ragService()
        rag_result = await asyncio.to_thread(
            rag_service.generate_context_rag_service,
            query=request.query,
            n_results=25,   # 넉넉히 뽑아 알레르기 필터 후에도 후보 확보
            exclude_allergies=request.session.allergies,
        )
        context = rag_result.get("context", "")
        logger.info(f"RAG 검색 완료 | context 길이: {len(context)}")

        # 2. context 채워서 LLMRequest 업데이트
        request.context = context

        # 3. LLM 추론도 스레드풀에서 실행 (GPU 연산, 무거운 작업)
        result = await asyncio.to_thread(
            LLMService.generate_result,
            session=request.session,
            query=request.query,
            context=request.context,
        )
        return result

    except Exception as e:
        logger.error(f"LLM 처리 오류: {str(e)} | 타입: {type(e).__name__}")
        logger.error(f"상세 위치:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))