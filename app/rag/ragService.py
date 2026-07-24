"""
RAG Service - Retriever + LLM 통합 모듈 (유사도 필터링 버전)
"""

from typing import Dict, Optional
import logging

# ✅ retriever.py의 싱글톤 함수명 규칙 유지
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


class RagService:
    """RAG Service - 검색 + LLM 답변 생성"""
    
    def __init__(self):
        """RAG Service 초기화"""
        self.retriever = get_retriever()
        logger.info("✅ RAG Service 초기화 완료")
    
    def search_rag_service(
        self,
        query: str,
        n_results: int = 3
    ) -> Dict:
        """RAG 기반 검색 (문서 검색만)"""
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
        n_results: int = 3,
        exclude_allergies: list = None,   # ← 파라미터 추가
    ) -> Dict:
        try:
            # 제외 재료가 있으면 검색 쿼리에서 그 단어+부정표현을 제거 —
            #   RAG(의미검색)가 "우유 없는"의 "우유"로 편향 검색하는 것 방지
            search_query = query
            if exclude_allergies:
                import re
                for _a in exclude_allergies:
                    search_query = search_query.replace(_a, " ")
                for _neg in ("없는", "없이", "안 들어간", "안들어간", "빼고", "빼", "말고", "제외"):
                    search_query = search_query.replace(_neg, " ")
                search_query = re.sub(r"\s+", " ", search_query).strip() or query
            search_result = self.search_rag_service(search_query, n_results)
            if not search_result["success"]:
                return search_result
            results = search_result["results"]
            THRESHOLD = 0.6
            document_texts = []
            filtered_results = []
            if results:
                filtered_idx = 1
                for result in results:
                    score = result.get('score', 0.0)
                    if score < THRESHOLD:
                        logger.info(f"⚠️ 유사도 점수 낮음 무시 ({score} < {THRESHOLD}): {result['menu_name']}")
                        continue
                    # ★ 알레르기 필터링 (추가된 부분)
                    if exclude_allergies:
                        doc_text_lower = result['document']
                        matched = [a for a in exclude_allergies if a in doc_text_lower]
                        if matched:
                            logger.warning(f"🚫 알레르기({matched}) 포함으로 제외: {result['menu_name']}")
                            continue
                    doc_text = f"""[메뉴 {filtered_idx}]
메뉴명: {result['menu_name']}
카테고리: {result['category']}
유사도: {score}
{result['document']}
"""
                    document_texts.append(doc_text)
                    filtered_results.append(result)
                    filtered_idx += 1
            if not document_texts:
                context = "[안내] 사용자의 요청과 관련된 메뉴 정보를 메뉴판에서 찾을 수 없습니다. 사용자에게 매장 메뉴판에 없는 메뉴라고 정중히 안내하세요."
                documents = []
            else:
                context = "\n".join(document_texts)
                documents = filtered_results
            return {
                "success": True,
                "query": query,
                "context": context,
                "documents": documents,
                "message": f"{len(documents)}개의 유효한 관련 메뉴 정보"
            }
        except Exception as e:
            logger.error(f"❌ 컨텍스트 생성 오류: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_service_info_rag_service(self) -> Dict:
        """RAG Service 정보 반환"""
        return {
            "service": "RAG Service",
            "status": "ready",
            "retriever_info": self.retriever.get_retriever_info_ret_rag_retriever()
        }


# 싱글톤 패턴 유지
_rag_service: Optional[RagService] = None


def get_rag_ragService() -> RagService:
    """RAG Service 인스턴스 반환"""
    global _rag_service
    
    if _rag_service is None:
        _rag_service = RagService()
    
    return _rag_service