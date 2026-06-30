"""
쿼리 텍스트 임베딩 모듈 - Multilingual-E5-Large
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """임베딩 모델 관리 클래스 - Multilingual-E5-Large"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: 사용할 모델명
                - "intfloat/multilingual-e5-large" (권장)
        
        주의: Multilingual-E5-Large는 프리픽스가 필수입니다!
        - 쿼리: "query: " 프리픽스 추가
        - 문서: "passage: " 프리픽스 추가
        """
        self.model_name = model_name
        self.query_prefix = "query: "      # 쿼리용 프리픽스
        self.passage_prefix = "passage: "  # 문서용 프리픽스
        
        try:
            logger.info(f"📥 임베딩 모델 로드 중: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ 모델 로드 완료: {model_name}")
            logger.info(f"💡 프리픽스 설정: query='{self.query_prefix}', passage='{self.passage_prefix}'")
        
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            raise
    
    def embed_query_em_rag_embedding(
        self,
        query: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        쿼리 텍스트를 임베딩 벡터로 변환
        
        Args:
            query: 쿼리 텍스트 또는 텍스트 리스트
        
        Returns:
            임베딩 벡터 또는 벡터 리스트
        
        Note:
            Multilingual-E5-Large는 쿼리에 "query: " 프리픽스가 필요합니다!
        
        Example:
            # 단일 쿼리
            embedding = model.embed_query_em_rag_embedding("아메리카노")
            # 내부: "query: 아메리카노" 로 임베딩됨
            
            # 여러 쿼리
            embeddings = model.embed_query_em_rag_embedding(
                ["아메리카노", "카페라떼"]
            )
        """
        try:
            # 단일 문자열인 경우 리스트로 변환
            if isinstance(query, str):
                query_list = [query]
                single = True
            else:
                query_list = query
                single = False
            
            # ✅ 쿼리에 프리픽스 추가 (E5-Large 필수!)
            prefixed_queries = [f"{self.query_prefix}{q}" for q in query_list]
            
            logger.debug(f"🔄 쿼리 임베딩 중: {len(query_list)}개...")
            logger.debug(f"  원본: {query_list[0]}")
            logger.debug(f"  변환: {prefixed_queries[0]}")
            
            # 임베딩 생성
            embeddings = self.model.encode(
                prefixed_queries,
                convert_to_numpy=False  # 리스트로 반환
            )
            
            # 단일 쿼리인 경우 첫 번째 요소만 반환
            if single:
                return embeddings[0]
            else:
                return embeddings
        
        except Exception as e:
            logger.error(f"❌ 쿼리 임베딩 오류: {str(e)}")
            raise
    
    def embed_documents_em_rag_embedding(
        self,
        documents: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        문서 텍스트를 임베딩 벡터로 변환
        
        Args:
            documents: 문서 텍스트 또는 문서 리스트
        
        Returns:
            임베딩 벡터 또는 벡터 리스트
        
        Note:
            Multilingual-E5-Large는 문서에 "passage: " 프리픽스가 필요합니다!
        
        Example:
            # 단일 문서
            embedding = model.embed_documents_em_rag_embedding("메뉴명: 아메리카노...")
            # 내부: "passage: 메뉴명: 아메리카노..." 로 임베딩됨
        """
        try:
            # 단일 문자열인 경우 리스트로 변환
            if isinstance(documents, str):
                doc_list = [documents]
                single = True
            else:
                doc_list = documents
                single = False
            
            # ✅ 문서에 프리픽스 추가 (E5-Large 필수!)
            prefixed_docs = [f"{self.passage_prefix}{doc}" for doc in doc_list]
            
            logger.debug(f"🔄 문서 임베딩 중: {len(doc_list)}개...")
            
            # 임베딩 생성
            embeddings = self.model.encode(
                prefixed_docs,
                convert_to_numpy=False  # 리스트로 반환
            )
            
            # 단일 문서인 경우 첫 번째 요소만 반환
            if single:
                return embeddings[0]
            else:
                return embeddings
        
        except Exception as e:
            logger.error(f"❌ 문서 임베딩 오류: {str(e)}")
            raise
    
    def get_model_info_em_rag_embedding(self) -> dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "model_type": "Multilingual-E5-Large",
            "embedding_dimension": self.model.get_embedding_dimension(),
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix
        }


# 싱글톤 패턴 (앱 시작 시 한 번만 로드)
_embedding_model: EmbeddingModel = None


def get_embedding_model_em_rag_embedding() -> EmbeddingModel:
    """임베딩 모델 인스턴스 반환"""
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name="intfloat/multilingual-e5-large")
    
    return _embedding_model