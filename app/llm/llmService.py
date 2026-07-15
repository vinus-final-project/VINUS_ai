import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import settings
from app.interface.dto.llmRequest import SessionState
from app.interface.dto.llmResult import LLMResult
from app.llm.promptBuilder import build_prompt_promptBuilder
from app.llm.outputParser import parse_llm_output_outputParser
from app.interface.dto.llmRequest import LLMRequest
import logging

logger = logging.getLogger(__name__)


class LLMService:

    _model = None
    _tokenizer = None

    # ------------------------------------------------------------------
    # 모델 초기화 (서버 시작 시 1회 호출)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_llmService(cls) -> None:
        """EXAONE 원본 모델 로드 (서버 시작 시 1회)"""
        if cls._model is not None:
            logger.info("LLM 모델 이미 로드됨 (스킵)")
            return

        logger.info(f"LLM 모델 로드 시작: {Settings.llm_model_path}")

        cls._tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model_path,
            trust_remote_code=True,
        )

        cls._model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
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
        request = LLMRequest(session=session, query=query, context=context)
        messages = build_prompt_promptBuilder(request)
        logger.info(f"프롬프트 생성 완료 | query: {query}")

        # 2. 토크나이즈
        input_ids = cls._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # 3. 생성
        with torch.no_grad():
            output_ids = cls._model.generate(
                input_ids,
                max_new_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                do_sample=True,
                eos_token_id=cls._tokenizer.eos_token_id,
            )

        # 4. 디코딩
        generated = output_ids[0][input_ids.shape[-1]:]
        raw_output = cls._tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"EXAONE 출력: {raw_output}")

        # 5. 파싱 → LLMResult 반환
        return parse_llm_output_outputParser(raw_output)