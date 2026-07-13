from llama_cpp import Llama
from app.core.config import Settings
from app.interface.dto.llmRequest import SessionState
from app.interface.dto.llmResult import LLMResult
from app.llm.promptBuilder import build_prompt_promptBuilder
from app.llm.outputParser import parse_llm_output_outputParser
from app.interface.dto.llmRequest import LLMRequest
import logging

logger = logging.getLogger(__name__)


class LLMService:

    
    _model: Llama = None

    # ------------------------------------------------------------------
    # 모델 초기화 (서버 시작 시 1회 호출)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_llmService(cls) -> None:
        """EXAONE GGUF 모델 로드 (서버 시작 시 1회)"""
        if cls._model is not None:
            logger.info("LLM 모델 이미 로드됨 (스킵)")
            return

        logger.info(f"LLM 모델 로드 시작: {Settings.llm_model_path}")

        
        cls._model = Llama(
            model_path=Settings.llm_model_path,
            n_ctx=3072,          # 컨텍스트 길이
            n_gpu_layers=35,      # 테스트용 CPU (EC2 배포 시 -1로 변경)
            verbose=False,
        )
        
        logger.info("LLM 모델 로드 완료")

    # ------------------------------------------------------------------
    # LLM 호출
    # ------------------------------------------------------------------
    @classmethod
    def generate_result(
        cls,
        session: SessionState,
        query: str,
        context: str,
    ) -> LLMResult:
        """Session, Query, Context → LLMResult 반환"""

        if cls._model is None:
            logger.error("LLM 모델 미초기화 상태")
            return LLMResult(
                result="",
                events=[],
                response="서비스 준비 중입니다. 잠시 후 다시 시도해주세요.",
                source="LLM"
            )

        # 1. 프롬프트 생성
        
        request = LLMRequest(session=session, query=query, context=context)
        messages = build_prompt_promptBuilder(request)
        logger.info(f"프롬프트 생성 완료 | query: {query}")

        # 2. EXAONE GGUF 호출
        response = cls._model.create_chat_completion(
            messages=messages,
            max_tokens=Settings.llm_max_tokens,
            temperature=Settings.llm_temperature,
        )

        # 3. 출력 추출
        raw_output = response["choices"][0]["message"]["content"]
        logger.info(f"EXAONE 출력: {raw_output}")

        # 4. 파싱 → LLMResult 반환
        return parse_llm_output_outputParser(raw_output)