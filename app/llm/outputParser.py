import json
import re
from app.interface.dto.llmResult import LLMResult, FSMEvent
import logging

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """EXAONE 출력에서 JSON 부분만 추출"""
    # JSON 블록이 ```json ... ``` 형식으로 감싸져 있을 경우 제거
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # 중괄호로 시작하는 JSON 직접 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_events(events_data: list) -> list[FSMEvent]:
    """이벤트 리스트 파싱 → FSMEvent 리스트 변환"""
    events = []
    for event in events_data:
        try:
            events.append(FSMEvent(
                type=event.get("type", ""),
                parameters=event.get("parameters", {})
            ))
        except Exception as e:
            logger.warning(f"이벤트 파싱 실패 (스킵): {event} | 오류: {str(e)}")
            continue
    return events


def parse_llm_output_outputParser(raw_output: str) -> LLMResult:
    """
    EXAONE 출력 텍스트 → LLMResult 변환
    파싱 실패 시 fallback LLMResult 반환
    """
    try:
        # JSON 추출 및 파싱
        json_str = _extract_json(raw_output)
        parsed = json.loads(json_str)

        response = parsed.get("response", "")
        events = _parse_events(parsed.get("events", []))

        logger.info(f"LLM 출력 파싱 완료 | response: {response} | events: {len(events)}개")

        return LLMResult(
            result=raw_output,   # LLM 원문 그대로
            events=events,
            response=response,
            source="LLM"
        )

    except json.JSONDecodeError as e:
        # JSON 파싱 실패 → fallback
        logger.warning(f"JSON 파싱 실패 | 오류: {str(e)} | 원문: {raw_output}")
        return LLMResult(
            result=raw_output,
            events=[],
            response="죄송합니다. 응답을 처리하는 중 오류가 발생했습니다.",
            source="LLM"
        )

    except Exception as e:
        # 그 외 오류 → fallback
        logger.error(f"출력 파싱 오류: {str(e)}")
        return LLMResult(
            result="",
            events=[],
            response="죄송합니다. 일시적인 오류가 발생했습니다.",
            source="LLM"
        )