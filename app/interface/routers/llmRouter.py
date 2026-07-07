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
        # 1. RAG 검색 → context 생성
        rag_service = get_rag_ragService()
        rag_result = rag_service.generate_context_rag_service(
            query=request.query,
            n_results=5
        )
        context = rag_result.get("context", "")
        logger.info(f"RAG 검색 완료 | context 길이: {len(context)}")

        # 2. context 채워서 LLMRequest 업데이트
        request.context = context

        # TODO: llmService 구현 완료 후 주석 해제
       
        result = LLMService.generate_result(
            session=request.session,
            query=request.query,
            context=request.context,
        )
        return result

        

    except Exception as e:
        logger.error(f"LLM 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))