"""
Chroma Vector DB 관리 모듈
"""

import chromadb
from typing import List, Dict, Optional
from pathlib import Path
import logging

from app.core.config import Settings  # ← 추가!

logger = logging.getLogger(__name__)


class ChromaManager:
    """Vector DB (Chroma) 관리 클래스"""
    
    def __init__(self, db_path: Path = None):
        """
        Chroma 클라이언트 초기화
        
        Args:
            db_path: Vector DB 저장 경로 (Path 객체 또는 문자열)
        """
        # ✅ Path 객체로 변환
        if db_path is None:
            self.db_path = Settings.chroma_db_path
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path
        
        self.collection_name = Settings.chroma_collection_name
        
        # ✅ str로 명시적 변환
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # ← 추가: 코사인 거리로 명시 지정 (기본값 L2 대신)
        )
        logger.info(f"✅ Chroma 클라이언트 초기화 완료: {self.db_path}")
    
    def search_cm_rag_chromaManager(
        self,
        query_embeddings: List[float],
        n_results: int = 3
    ) -> Dict:
        """
        임베딩 벡터로 유사 문서 검색
        
        Args:
            query_embeddings: 쿼리 임베딩 벡터
            n_results: 반환할 결과 개수
        
        Returns:
            dict: 검색 결과
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            
            logger.debug(f"✅ 검색 완료: {len(results['ids'][0])}개 결과")
            return {
                "success": True,
                "data": results
            }
        
        except Exception as e:
            logger.error(f"❌ 검색 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_collection_info_cm_rag_chromaManager(self) -> Dict:
        """
        컬렉션 정보 조회
        
        Returns:
            dict: 컬렉션 정보
        """
        try:
            count = self.collection.count()
            logger.info(f"📊 컬렉션 정보: {count}개 문서")
            
            return {
                "success": True,
                "count": count,
                "collection_name": self.collection_name,  # ← 수정: 하드코딩된 "menu_collection" → self.collection_name ("vinus_menus")
                "db_path": str(self.db_path)  # ✅ Path → str 변환
            }
        
        except Exception as e:
            logger.error(f"❌ 컬렉션 조회 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_by_id_cm_rag_chromaManager(self, doc_id: str) -> Optional[Dict]:
        """
        ID로 문서 조회
        
        Args:
            doc_id: 문서 ID
        
        Returns:
            dict: 문서 정보 또는 None
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result['ids']:
                return {
                    "success": True,
                    "id": result['ids'][0],
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
            else:
                return {
                    "success": False,
                    "message": f"문서 {doc_id}를 찾을 수 없습니다"
                }
        
        except Exception as e:
            logger.error(f"❌ 문서 조회 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# 싱글톤 패턴 (앱 전체에서 하나만 사용)
_chroma_manager: Optional[ChromaManager] = None


def get_chroma_manager_cm_rag_chromaManager() -> ChromaManager:
    """Chroma 매니저 인스턴스 반환"""
    global _chroma_manager
    
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    
    return _chroma_manager