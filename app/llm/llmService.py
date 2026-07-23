import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import Settings
from app.interface.dto.llmRequest import LLMRequest, SessionState
from app.interface.dto.llmResult import LLMResult
from app.llm.outputParser import parse_llm_output_outputParser
from app.llm.promptBuilder import build_prompt_promptBuilder

logger = logging.getLogger(__name__)

  # 커밋용

class LLMService:

    _model = None
    _tokenizer = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 감지

    # ------------------------------------------------------------------
    # 모델 초기화 (서버 시작 시 1회 호출)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_llmService(cls) -> None:
        """EXAONE 원본 모델 로드 (서버 시작 시 1회)"""
        if cls._model is not None:
            logger.info("LLM 모델 이미 로드됨 (스킵)")
            return

        # Settings 인스턴스화
        settings = Settings()
        
        # GPU 사용 여부 체크
        if not torch.cuda.is_available():
            logger.critical("🚫 [경고] GPU(CUDA)를 사용할 수 없습니다! CPU 모드로 동작하거나 에러가 발생할 수 있습니다.")
            cls._device = "cpu"

        logger.info(f"LLM 모델 로드 시작 ({cls._device} 모드): {settings.llm_model_path}")

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model_path,
                trust_remote_code=True,
            )

            # GPU 가용 여부에 따라 device_map 설정
            device_map_config = "cuda" if torch.cuda.is_available() else "cpu"

            cls._model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map_config,
                trust_remote_code=True,
            )
            cls._model.eval()

            logger.info("✅ LLM 모델 로드 완료")

        except Exception as e:
            logger.error(f"❌ LLM 모델 로드 중 치명적 오류 발생: {str(e)}")
            raise e  # main.py의 lifespan에서 이 에러를 캐치하게 함

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

        settings = Settings()

        # 1. 프롬프트 생성
        request = LLMRequest(session=session, query=query, context=context)
        messages = build_prompt_promptBuilder(request)
        logger.info(f"프롬프트 생성 완료 | query: {query}")

        # 2. 토크나이즈 및 디바이스 할당
        inputs = cls._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(cls._device)

        # 3. 생성
        with torch.no_grad():
            output_ids = cls._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                do_sample=True,
                eos_token_id=cls._tokenizer.eos_token_id,
            )

        # 4. 디코딩
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_output = cls._tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"EXAONE 출력: {raw_output}")

        # 5. 파싱 → LLMResult 반환
        return parse_llm_output_outputParser(raw_output)