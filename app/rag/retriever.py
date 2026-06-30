"""
Vector Search를 통한 문서 검색 모듈
"""

from typing import List, Dict, Optional
import logging

from app.rag.embedding import get_embedding_model_em_rag_embedding
from app.rag.chromaManager import get_chroma_manager_cm_rag_chromaManager

logger = logging.getLogger(__name__)


class Retriever:
    """Vector Search 기반 문서 검색 클래스"""
    
    def __init__(self):
        """Retriever 초기화"""
        self.embedding_model = get_embedding_model_em_rag_embedding()
        self.chroma_manager = get_chroma_manager_cm_rag_chromaManager()
        logger.info("✅ Retriever 초기화 완료")
    
    def retrieve_ret_rag_retriever(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict:
        """
        쿼리로 유사한 메뉴 문서 검색
        
        Args:
            query: 사용자 질문
            n_results: 반환할 결과 개수
        
        Returns:
            dict: 검색 결과
                {
                    "success": bool,
                    "query": str,
                    "results": [
                        {
                            "id": str,
                            "menu_name": str,
                            "document": str,
                            "category": str,
                            "score": float
                        },
                        ...
                    ]
                }
        
        Example:
            retriever = Retriever()
            result = retriever.retrieve_ret_rag_retriever(
                "에스프레소 들어간 음료",
                n_results=3
            )
        """
        try:
            logger.info(f"🔍 검색 시작: '{query}'")
            
            # [1] 쿼리 임베딩
            logger.debug(f"📊 쿼리 임베딩 중...")
            query_embedding = self.embedding_model.embed_query_em_rag_embedding(query)
            
            # [2] Vector DB 검색
            logger.debug(f"🔎 Vector DB 검색 중...")
            search_result = self.chroma_manager.search_cm_rag_chromaManager(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "message": "검색 중 오류 발생"
                }
            
            # [3] 결과 정리
            chroma_data = search_result["data"]
            
            results = []
            if chroma_data["ids"] and len(chroma_data["ids"]) > 0:
                for idx, doc_id in enumerate(chroma_data["ids"][0]):
                    # 거리를 유사도 점수로 변환 (0~1, 1이 가장 유사)
                    distance = chroma_data["distances"][0][idx]
                    similarity_score = 1 - distance  # Cosine distance → similarity
                    
                    results.append({
                        "id": doc_id,
                        "menu_name": chroma_data["metadatas"][0][idx].get("menu_name"),
                        "category": chroma_data["metadatas"][0][idx].get("category"),
                        "document": chroma_data["documents"][0][idx],
                        "score": round(similarity_score, 4)  # 소수점 4자리
                    })
            
            logger.info(f"✅ 검색 완료: {len(results)}개 결과")
            
            return {
                "success": True,
                "query": query,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"❌ 검색 오류: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_retriever_info_ret_rag_retriever(self) -> Dict:
        """Retriever 정보 반환"""
        collection_info = self.chroma_manager.get_collection_info_cm_rag_chromaManager()
        model_info = self.embedding_model.get_model_info_em_rag_embedding()
        
        return {
            "retriever": "Vector Search Retriever",
            "model": model_info,
            "collection": collection_info
        }


# 싱글톤 패턴
_retriever: Optional[Retriever] = None


def get_retriever_ret_rag_retriever() -> Retriever:
    """Retriever 인스턴스 반환"""
    global _retriever
    
    if _retriever is None:
        _retriever = Retriever()
    
    return _retriever