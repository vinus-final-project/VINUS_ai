from app.rag.ragDocuments import import_csv_to_vectordb_rag_ragDocuments
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 서버 시작...")
    
    try:
        # CSV에서 Vector DB로 로드
        result = import_csv_to_vectordb_rag_ragDocuments("./data/rag_documents.csv")
        
        if result["success"]:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ {result['message']}")
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
    
    yield
    print("👋 서버 종료")

app = FastAPI(lifespan=lifespan)