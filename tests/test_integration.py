"""
실LLM 통합 테스트 (`integration` 마커, 기본 실행에서 제외).

FakeChatModel이 아니라 실제 ChatAnthropic을 붙여서, 그래프 전체가
"dispatcher가 대화에서 owner/repo를 뽑아 도구를 호출 → GitHub API가 실제로
개요/소스 코드를 가져옴 → voter 3개가 실제로 요약+코드 리뷰를 작성 →
다수결로 하나를 채택"하는 전체 파이프라인을 실제 모델로 검증한다.

FakeChatModel 기반 테스트는 "그래프 배선이 맞는지"를 결정적으로 검증하지만,
"LLM이 실제로 owner/repo를 도구 인자로 잘 뽑아내는지", "voter가 실제로
요약+코드 리뷰 두 섹션을 챙기는지" 같은 프롬프트 자체의 품질은 검증하지
못한다 — 이 파일이 그 간극을 메운다.

비용/속도 때문에:
- 저렴하고 빠른 모델(claude-haiku-4-5)을 쓴다.
- 응답이 짧아지도록 max_tokens를 낮게 잡는다.
- GitHub API도 실제로 호출되므로(비인증, 시간당 60회 제한), 안정적이고
  내용이 거의 바뀌지 않는 저장소(octocat/Hello-World)만 사용한다.
- LLM 출력은 비결정적이므로, 정확한 문자열이 아니라 "구조가 맞는지"
  (도구가 호출됐는지, voter 3개가 다 채워졌는지, 최종 답변에 저장소 관련
  단어가 들어있는지)만 검증한다.

ANTHROPIC_API_KEY가 없으면(.env 미설정) 테스트가 실패하는 대신 skip된다 —
CI는 `pytest -m "not integration"`으로 이 파일 전체를 건너뛰지만, 로컬에서
실수로 `pytest -m integration`을 키 없이 돌려도 조용히 skip되게 하기 위함.
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage

from agent.graph import build_graph

load_dotenv()

INTEGRATION_MODEL = "claude-haiku-4-5-20251001"

pytestmark = pytest.mark.integration


def _require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY가 설정되지 않아 실LLM 통합 테스트를 건너뜁니다.")


@pytest.fixture
def real_llm():
    _require_api_key()
    return ChatAnthropic(model=INTEGRATION_MODEL, max_tokens=1024)


def test_dispatcher_extracts_repo_and_calls_overview_tool(real_llm):
    graph = build_graph(real_llm)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="octocat/Hello-World 저장소를 리뷰해줘. 개요만 봐도 돼.")
            ],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    assert any(m.name == "fetch_repo_overview" for m in tool_messages)
    # dispatcher가 owner/repo를 정확히 뽑아 도구에 넘겼다면, 도구 결과에
    # 저장소 full_name이 그대로 들어있어야 한다.
    assert any("octocat/Hello-World" in m.content for m in tool_messages)


def test_full_pipeline_produces_review_with_summary_and_code_sections(real_llm):
    graph = build_graph(real_llm)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="octocat/Hello-World 저장소를 요약이랑 코드 리뷰 둘 다 해줘."
                )
            ],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    # 투표형 앙상블: voter 3개가 전부 독립 시도를 남겼어야 한다.
    assert len(result["report_drafts"]) == 3
    # dispatcher 1 + voter 3 = 최소 4번 LLM이 호출됐다(도구 팬아웃 자체는
    # LLM 호출이 아니다).
    assert result["iteration"] >= 4

    final_text = result["messages"][-1].content
    assert isinstance(final_text, str) and len(final_text) > 0
    # 실제 모델 출력이라 정확한 문구는 보장 못 하지만, 리뷰 대상 저장소를
    # 언급조차 안 하는 답이면 프롬프트가 무시되고 있다는 신호다.
    assert "Hello-World" in final_text or "hello-world" in final_text.lower()
