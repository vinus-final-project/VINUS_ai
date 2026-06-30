"""
RAG Service - Retriever + LLM 통합 모듈
"""

from typing import Dict, Optional
import logging

from app.rag.retriever import get_retriever_ret_rag_retriever

logger = logging.getLogger(__name__)


class RagService:
    """RAG Service - 검색 + LLM 답변 생성"""
    
    def __init__(self):
        """RAG Service 초기화"""
        self.retriever = get_retriever_ret_rag_retriever()
        logger.info("✅ RAG Service 초기화 완료")
    
    def search_rag_service(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict:
        """
        RAG 기반 검색 (문서 검색만)
        
        Args:
            query: 사용자 질문
            n_results: 반환할 결과 개수
        
        Returns:
            dict: 검색 결과
        
        Example:
            service = RAGService()
            result = service.search_rag_service_rag_ragService(
                "에스프레소 들어간 음료",
                n_results=3
            )
        """
        try:
            logger.info(f"🔍 RAG 검색 시작: '{query}'")
            
            # Retriever로 검색
            search_result = self.retriever.retrieve_ret_rag_retriever(
                query=query,
                n_results=n_results
            )
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "message": search_result.get("message", "검색 실패")
                }
            
            logger.info(f"✅ 검색 완료: {len(search_result['results'])}개 결과")
            
            return {
                "success": True,
                "query": query,
                "results": search_result["results"]
            }
        
        except Exception as e:
            logger.error(f"❌ RAG 검색 오류: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def generate_context_rag_service(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict:
        """
        검색 결과로부터 LLM 입력용 컨텍스트 생성
        
        Args:
            query: 사용자 질문
            n_results: 검색 결과 개수
        
        Returns:
            dict: LLM에 전달할 컨텍스트
                {
                    "success": bool,
                    "query": str,
                    "context": str (LLM 입력용),
                    "documents": [...],
                    "message": str (선택사항)
                }
        """
        try:
            # 검색 실행
            search_result = self.search_rag_service(query, n_results)
            
            if not search_result["success"]:
                return search_result
            
            # 컨텍스트 생성
            results = search_result["results"]
            
            if not results:
                context = "검색된 메뉴 정보가 없습니다."
                documents = []
            else:
                # 문서들을 포맷팅
                document_texts = []
                for idx, result in enumerate(results, 1):
                    doc_text = f"""[메뉴 {idx}]
메뉴명: {result['menu_name']}
카테고리: {result['category']}
유사도: {result['score']}

{result['document']}
"""
                    document_texts.append(doc_text)
                
                context = "\n".join(document_texts)
                documents = results
            
            logger.info(f"✅ 컨텍스트 생성 완료")
            
            return {
                "success": True,
                "query": query,
                "context": context,
                "documents": documents,
                "message": f"{len(results)}개의 관련 메뉴 정보"
            }
        
        except Exception as e:
            logger.error(f"❌ 컨텍스트 생성 오류: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_service_info_rag_service(self) -> Dict:
        """RAG Service 정보 반환"""
        return {
            "service": "RAG Service",
            "status": "ready",
            "retriever_info": self.retriever.get_retriever_info_ret_rag_retriever()
        }


# 싱글톤 패턴
_rag_service: Optional[RagService] = None


def get_rag_service_rag_service() -> RagService:
    """RAG Service 인스턴스 반환"""
    global _rag_service
    
    if _rag_service is None:
        _rag_service = RagService()
    
    return _rag_service