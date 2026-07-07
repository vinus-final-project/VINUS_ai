import os
from pathlib import Path

class Settings:
    # 📂 기본 경로 설정 (C:\Final project\VINUS_ai)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # 🖥️ 서버 설정
    app_name: str = "VINUS"
    app_host: str = "0.0.0.0"
    app_port: int = 8001

    # 🤖 LLM 설정
    llm_model_name: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7

    # 🗄️ ChromaDB 설정
    chroma_db_path: Path = BASE_DIR / "rag_db"
    chroma_collection_name: str = "vinus_menus"
    
    # 🌟 RAG CSV 파일 경로
    rag_documents_csv_path: Path = BASE_DIR / "data" / "rag_documents.csv"

    # 🔤 Embedding 설정
    embedding_model_name: str = "intfloat/multilingual-e5-large"


# 🌟 외부 파일들이 이 settings 객체를 참조하게 됩니다.
settings = Settings()