from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # 서버 설정
    app_name: str = "VINUS"
    app_host: str = "0.0.0.0" # "모든 IP에서 접속 허용"이라는 뜻
    app_port: int = 8001 # 이건 FastAPI 서버가 실행될 포트 번호 (백엔드가 8000번에서 실행하면 충돌 방지를 위해 8001번으로 설정)

    # LLM 설정
    llm_model_path: str = "./models/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"  # GGUF 파일 경로 (테스트용)
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # ChromaDB 설정
    chroma_db_path: Path = Path("./rag_db")  # RAG 담당자 ChromaDB 데이터베이스 파일 경로
    chroma_collection_name: str = "vinus_menus"
    chroma_db_dir: str = "./chroma_db" 
    # RAG 문서 설정 (RAG 담당자 필드명 맞춤)
    rag_documents_csv_path: str = "./data/rag_documents.csv"  # RAG 담당자 추가

    # Embedding 설정
    embedding_model_name: str = "intfloat/multilingual-e5-large"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()