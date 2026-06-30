"""
기존 RAG 문서 CSV를 Vector DB에 로드하는 모듈
"""

import pandas as pd
from pathlib import Path
from typing import Dict

# ✅ 우리가 만들어 둔 싱글톤 모델 및 매니저 임포트
from app.rag.embedding import get_embedding_model_em_rag_embedding
from app.rag.chromaManager import get_chroma_manager_cm_rag_chromaManager

def import_csv_to_vectordb_rag_ragDocuments(csv_path: str = "./data/rag_documents.csv") -> Dict:
    """
    CSV 파일의 RAG 문서들을 우리가 설정한 임베딩 모델을 통해 Vector DB에 로드
    """
    try:
        print("\n" + "="*70)
        print("📥 RAG 문서 CSV 로드 시작")
        print("="*70)
        
        # [1] CSV 파일 읽기
        if not Path(csv_path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
            return {"success": False, "message": "파일을 찾을 수 없습니다"}
        
        print(f"\n📄 CSV 파일 읽기: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✅ {len(df)}개 문서 읽음")
        
        # [2] 프로젝트 공통 싱글톤 인스턴스 로드
        print("\n🔧 Embedding 모델 및 Vector DB 초기화...")
        embedding_model = get_embedding_model_em_rag_embedding()
        chroma_manager = get_chroma_manager_cm_rag_chromaManager()
        
        # chromaManager 내부에서 사용하는 원본 컬렉션 객체 획득
        # (만약 chromaManager에 collection 속성이 다른 이름이라면 구조에 맞게 수정 필요)
        collection = chroma_manager.collection
        
        # [3] 기존 문서 삭제 (선택사항 - 중복 방지 필요시 주석 해제)
        # collection.delete(where={})
        
        # [4] CSV 데이터를 리스트로 변환
        ids = []
        documents = []
        metadatas = []
        
        for idx, row in df.iterrows():
            ids.append(str(row['doc_id']))  # ChromaID는 문자열이 안전합니다.
            documents.append(row['document'])
            metadatas.append({
                "menu_id": row['menu_id'],
                "menu_name": row['menu_name'],
                "category": row['category']
            })
            
        # [5] 🔥 핵심: 한국어 임베딩 벡터 일괄 생성
        print(f"\n📊 {len(df)}개 문서 임베딩 벡터 생성 중...")
        # embedding.py에 구현된 다중 문서 임베딩 메서드 호출 (메서드명이 다르면 매핑 필요)
        embeddings = embedding_model.embed_documents_em_rag_embedding(documents)
        
        # [6] Bulk insert (텍스트 + 임베딩 벡터 + 메타데이터 함께 적재)
        print(f"💾 Vector DB에 데이터 적재 중...")
        collection.add(
            ids=ids,
            embeddings=embeddings,  # ✅ 드디어 우리가 지정한 올바른 임베딩 벡터가 들어갑니다!
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"\n✅ Vector DB 로드 완료 ({len(ids)}개 문서)")
        print("="*70 + "\n")
        
        return {
            "success": True,
            "count": len(ids),
            "message": f"{len(ids)}개 RAG 문서를 올바른 임베딩 벡터와 함께 Vector DB에 로드했습니다"
        }
    
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    result = import_csv_to_vectordb_rag_ragDocuments()
    print(result)