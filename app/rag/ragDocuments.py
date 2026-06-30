"""
기존 RAG 문서 CSV를 Vector DB에 로드하는 모듈
"""

import chromadb
import pandas as pd
from pathlib import Path

def import_csv_to_vectordb_rag_ragDocuments(csv_path: str = "./data/rag_documents.csv"):
    """
    CSV 파일의 RAG 문서들을 Vector DB에 로드
    
    Args:
        csv_path: CSV 파일 경로
    
    Returns:
        dict: 로드 결과
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
        
        # [2] Vector DB 초기화
        print("\n🔧 Vector DB 초기화...")
        chroma_client = chromadb.PersistentClient(path="./rag_db")
        collection = chroma_client.get_or_create_collection(
            name="menu_collection"
        )
        
        # [3] 기존 문서 삭제 (선택사항 - 중복 방지)
        # collection.delete(where={})  # 주석 해제하면 기존 문서 모두 삭제
        
        # [4] CSV 데이터를 Vector DB에 로드
        print(f"\n💾 {len(df)}개 문서를 Vector DB에 로드 중...")
        
        ids = []
        documents = []
        metadatas = []
        
        for idx, row in df.iterrows():
            ids.append(row['doc_id'])
            documents.append(row['document'])
            metadatas.append({
                "menu_id": row['menu_id'],
                "menu_name": row['menu_name'],
                "category": row['category']
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  ├─ {idx + 1}/{len(df)} 처리 완료...")
        
        # Bulk insert
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"\n✅ Vector DB 로드 완료 ({len(ids)}개 문서)")
        print("="*70 + "\n")
        
        return {
            "success": True,
            "count": len(ids),
            "message": f"{len(ids)}개 RAG 문서를 Vector DB에 로드했습니다"
        }
    
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    result = import_csv_to_vectordb_rag_ragDocuments()
    print(result)