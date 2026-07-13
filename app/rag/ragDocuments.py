"""
기존 RAG 문서 CSV를 Vector DB에 로드하는 모듈
"""

import chromadb
import pandas as pd
from pathlib import Path
import logging

from app.rag.embedding import get_embedding_model_em_rag_embedding
from app.core.config import Settings  # ← 이미 있음

logger = logging.getLogger(__name__)


def import_csv_to_vectordb_rag_ragDocuments(csv_path: Path = None):
    """
    CSV 파일의 RAG 문서들을 Multilingual-E5-Large로 임베딩한 후 Vector DB에 로드
    
    Args:
        csv_path: CSV 파일 경로 (Path 객체 또는 문자열)
    """
    
    # ✅ Path 객체로 변환
    if csv_path is None:
        csv_path = Settings.rag_documents_csv_path  # ← 수정: 소문자 settings(미정의) → Settings로 변경
    elif isinstance(csv_path, str):
        csv_path = Path(csv_path)
    
    try:
        print("\n" + "="*70)
        print("📥 RAG 문서 CSV 로드 시작 (Multilingual-E5-Large 임베딩)")
        print("="*70)
        
        # [1] CSV 파일 읽기 ✅ (Path 객체 사용 가능)
        if not csv_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
            return {"success": False, "message": f"파일을 찾을 수 없습니다: {csv_path}"}
        
        print(f"\n📄 CSV 파일 읽기: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Path 객체 그대로 사용 가능
        
        print(f"✅ {len(df)}개 문서 읽음")
        
        # [2] 임베딩 모델 로드 (GPU 없으면 여기서 RuntimeError 발생)
        print(f"\n🤖 임베딩 모델 로드 (Multilingual-E5-Large)...")
        embedding_model = get_embedding_model_em_rag_embedding()
        print(f"✅ 모델 로드 완료")
        
        # [3] Vector DB 초기화 ✅ (Path 객체 사용)
        print(f"\n🔧 Vector DB 초기화...")
        print(f"   경로: {Settings.chroma_db_path}")
        
        chroma_client = chromadb.PersistentClient(
            path=str(Settings.chroma_db_path)  # ✅ 명시적으로 str 변환
        )
        collection = chroma_client.get_or_create_collection(
            name=Settings.chroma_collection_name
        )
        
        # [4] CSV 데이터를 임베딩한 후 Vector DB에 로드
        print(f"\n💾 {len(df)}개 문서를 임베딩 후 Vector DB에 로드 중...")
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for idx, row in df.iterrows():
            doc_text = row['document']
            doc_embedding = embedding_model.embed_documents_em_rag_embedding(doc_text)
            
            ids.append(row['doc_id'])
            documents.append(doc_text)
            embeddings.append(doc_embedding)
            metadatas.append({
                "menu_id": row['menu_id'],
                "menu_name": row['menu_name'],
                "category": row['category']
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  ├─ {idx + 1}/{len(df)} 임베딩 완료...")
        
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Vector DB 로드 완료 ({len(ids)}개 문서)")
        print("="*70 + "\n")
        
        return {
            "success": True,
            "count": len(ids),
            "message": f"{len(ids)}개 RAG 문서를 Multilingual-E5-Large로 임베딩하여 Vector DB에 로드했습니다"
        }
    
    except RuntimeError as e:
        # ← 추가: GPU(CUDA) 관련 에러는 삼키지 않고 그대로 다시 던져서 서버 기동 자체를 막음
        print(f"\n🚫 치명적 오류(GPU 미사용 환경): {str(e)}")
        logger.critical(f"🚫 GPU 필수 환경인데 사용 불가: {str(e)}")
        raise
    
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        logger.error(f"❌ 로드 오류: {str(e)}")
        return {"success": False, "message": str(e)}