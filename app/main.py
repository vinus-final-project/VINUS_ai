from app.rag.ragDocuments import import_csv_to_vectordb_rag_ragDocuments
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

# ✅ 로거 설정
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 생명주기"""
    
    # ✅ Startup
    logger.info("🚀 VINUS 서버 시작...")
    
    try:
        # CSV에서 Vector DB로 로드
        result = import_csv_to_vectordb_rag_ragDocuments("data/rag_documents.csv")
        
        if result["success"]:
            logger.info(f"✅ RAG 초기화 완료: {result['message']}")
        else:
            logger.warning(f"⚠️ RAG 초기화 실패: {result['message']}")
    
    except Exception as e:
        logger.error(f"❌ RAG 로드 오류: {str(e)}")
    
    yield
    
    # ✅ Shutdown
    logger.info("👋 VINUS 서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="VINUS",
    description="음성 기반 주문 시스템",
    lifespan=lifespan
)