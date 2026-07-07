from app.rag.ragDocuments import import_csv_to_vectordb_rag_ragDocuments
from app.rag.ragService import get_rag_ragService
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from app.core.config import settings
from app.llm.llmService import LLMService
import logging
from app.interface.routers.llmRouter import router as llm_router
# ✅ 로거 설정
logger = logging.getLogger(__name__)

# ========================================================================
# 📝 요청 모델
# ========================================================================

class STTQueryRequest(BaseModel):
    """STT 결과 검색 요청"""
    query: str
    n_results: int = 5

# ========================================================================
# 🔄 생명주기
# ========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 생명주기"""
    
    # ✅ Startup
    logger.info("🚀 VINUS 서버 시작...")
    
    try:
        result = import_csv_to_vectordb_rag_ragDocuments("data/rag_documents.csv")
        
        if result["success"]:
            logger.info(f"✅ RAG 초기화 완료: {result['message']}")
        else:
            logger.warning(f"⚠️ RAG 초기화 실패: {result['message']}")
    
    except Exception as e:
        logger.error(f"❌ RAG 로드 오류: {str(e)}")
    
    try:
        LLMService.initialize_llmService()
        logger.info("✅ LLM 모델 로드 완료")
    except Exception as e:
        logger.error(f"❌ LLM 로드 오류: {str(e)}")
    yield
    
    # ✅ Shutdown
    logger.info("👋 VINUS 서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    description="음성 기반 주문 시스템",
    lifespan=lifespan
)

# ========================================================================
# 🔍 검색 엔드포인트
# ========================================================================
# TODO: llmService 구현 완료 후 주석 해제
app.include_router(llm_router, tags=["llm"])

@app.post("/api/v1/rag/search")
async def search_rag(request: STTQueryRequest):
    """STT 결과로 RAG 검색"""
    logger.info(f"🔍 RAG 검색 요청: '{request.query}'")
    
    rag_service = get_rag_ragService()
    # ⚠️ 깨끗하게 정돈된 메서드명으로 수정 완료
    result = rag_service.search_rag_service(
        query=request.query,
        n_results=request.n_results
    )
    
    return result


@app.post("/api/v1/rag/search-with-context")
async def search_rag_with_context(request: STTQueryRequest):
    """STT 결과로 RAG 검색 + 컨텍스트 생성"""
    logger.info(f"🔍 RAG 검색 (컨텍스트 포함): '{request.query}'")
    
    rag_service = get_rag_ragService()
    result = rag_service.generate_context_rag_service(
        query=request.query,
        n_results=request.n_results
    )
    
    return result


@app.get("/api/v1/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": settings.app_name
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)

