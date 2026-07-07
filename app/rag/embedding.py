"""
쿼리 텍스트 임베딩 모듈 - Multilingual-E5-Large
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

from app.core.config import settings  # ← 추가!

logger = logging.getLogger(__name__)


class Embedding:
    """임베딩 모델 관리 클래스 - Multilingual-E5-Large"""
    
    def __init__(self, model_name: str = None):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: 사용할 모델명 (None이면 config에서 가져옴)
        """
        # ✅ config에서 모델명 가져옴
        self.model_name = model_name or settings.embedding_model_name
        self.query_prefix = "query: "
        self.passage_prefix = "passage: "
        
        try:
            logger.info(f"📥 임베딩 모델 로드 중: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ 모델 로드 완료: {self.model_name}")
            logger.info(f"💡 프리픽스 설정: query='{self.query_prefix}', passage='{self.passage_prefix}'")
        
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            raise
    
    def embed_query_em_rag_embedding(
        self,
        query: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """쿼리 텍스트를 임베딩 벡터로 변환"""
        try:
            if isinstance(query, str):
                query_list = [query]
                single = True
            else:
                query_list = query
                single = False
            
            prefixed_queries = [f"{self.query_prefix}{q}" for q in query_list]
            
            logger.debug(f"🔄 쿼리 임베딩 중: {len(query_list)}개...")
            
            embeddings = self.model.encode(
                prefixed_queries,
                convert_to_numpy=False
            )
            # 🌟 [추가] 만약 결과가 PyTorch 텐서 리스트라면 순수 파이썬 리스트로 변환합니다.
            if isinstance(embeddings, list):
                embeddings = [t.tolist() if hasattr(t, "tolist") else t for t in embeddings]
            elif hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            
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
        """문서 텍스트를 임베딩 벡터로 변환"""
        try:
            if isinstance(documents, str):
                doc_list = [documents]
                single = True
            else:
                doc_list = documents
                single = False
            
            prefixed_docs = [f"{self.passage_prefix}{doc}" for doc in doc_list]
            
            logger.debug(f"🔄 문서 임베딩 중: {len(doc_list)}개...")
            
            embeddings = self.model.encode(
                prefixed_docs,
                convert_to_numpy=False
            )
            
            # 🌟 [추가] 만약 결과가 PyTorch 텐서 리스트라면 순수 파이썬 리스트로 변환합니다.
            if isinstance(embeddings, list):
                embeddings = [t.tolist() if hasattr(t, "tolist") else t for t in embeddings]
            elif hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            
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


# 싱글톤 패턴
_embedding_model: Embedding = None


def get_embedding_model_em_rag_embedding() -> Embedding:
    """임베딩 모델 인스턴스 반환"""
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = Embedding()  # ✅ config에서 자동 읽음
    
    return _embedding_model