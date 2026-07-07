from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.core.config import settings
from app.interface.dto.llmRequest import LLMRequest, SessionState
from app.interface.dto.llmResult import LLMResult
from app.llm.promptBuilder import build_prompt_promptBuilder
from app.llm.outputParser import parse_llm_output_outputParser
import logging

logger = logging.getLogger(__name__)


class LLMService:

    _tokenizer = None
    _model = None

    # ------------------------------------------------------------------
    # 모델 초기화 (서버 시작 시 1회 호출)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_llmService(cls) -> None:
        """EXAONE 모델 로드 (서버 시작 시 1회)"""
        if cls._model is not None:
            logger.info("LLM 모델 이미 로드됨 (스킵)")
            return

        logger.info(f"LLM 모델 로드 시작: {settings.llm_model_name}")

        cls._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
        cls._model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_name,
            torch_dtype=torch.float16,   # fp16으로 로드
            device_map="auto",           # GPU 자동 할당
        )
        cls._model.eval()
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
        from app.interface.dto.llmRequest import LLMRequest
        request = LLMRequest(session=session, query=query, context=context)
        messages = build_prompt_promptBuilder(request)
        logger.info(f"프롬프트 생성 완료 | query: {query}")

        # 2. 토크나이징
        inputs = cls._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(cls._model.device)

        # 3. EXAONE 호출
        with torch.no_grad():
            outputs = cls._model.generate(
                inputs,
                max_new_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                do_sample=True,
                pad_token_id=cls._tokenizer.eos_token_id,
            )

        # 4. 출력 디코딩 (입력 토큰 제외)
        raw_output = cls._tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        )
        logger.info(f"EXAONE 출력: {raw_output}")

        # 5. 파싱 → LLMResult 반환
        return parse_llm_output_outputParser(raw_output)