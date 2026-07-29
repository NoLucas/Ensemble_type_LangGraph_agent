"""
그래프 통합(E2E) 테스트.

실제 도구(fetch_repo_overview, fetch_repo_source_sample)와 FakeChatModel을
결합해서 그래프 전체가 dispatcher -> [repo_overview_node, repo_source_node]
(도구 팬아웃) -> [voter_1, voter_2, voter_3] (투표형 앙상블 팬아웃) ->
vote_for_best_report -> END 순서로 정상 동작하는지 검증한다. LLM 자체는
가짜지만, GitHub API는 mock_github_get(conftest.py)으로 대체하고, 도구
실행과 상태 병합(add_messages/operator.add reducer)은 실제 코드 경로를
그대로 탄다.

가장 중요한 두 케이스:
1. test_graph_votes_for_the_candidate_that_matches_tool_facts — 세 voter가
   서로 다른 답을 내도(둘은 틀리고 하나만 도구 결과를 정확히 반영), 최종
   답변은 항상 정확한 후보여야 한다. 이게 깨지면 다수결 로직이 무너졌다는
   뜻이다.
2. test_graph_ignores_tool_calls_returned_by_voter_node — voter 노드는
   bind_tools되지 않은 llm을 쓰므로 실제로는 tool_call을 만들 수 없지만,
   FakeChatModel은 무엇이든 반환할 수 있다. voter 응답에 tool_call이 섞여
   있어도 그래프가 절대 실행하지 않는다는 것 — 안전장치가 "프롬프트 준수"가
   아니라 "그래프 구조(bind_tools 여부 + 고정 엣지)"라는 것을 검증한다.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from tests.conftest import FakeResponse


def test_graph_dispatches_overview_then_votes_for_report(fake_llm_factory, mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"full_name": "octocat/hello-world", "stargazers_count": 42}),
            FakeResponse(200, text="README"),
        ]
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_repo_overview", "args": {"repo": "octocat/hello-world"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="이 저장소는 stars: 42를 받았습니다."),
        AIMessage(content="이 저장소는 stars: 42를 받았습니다."),
        AIMessage(content="이 저장소는 stars: 42를 받았습니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="octocat/hello-world 리뷰해줘")],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    assert result["messages"][-1].content == "이 저장소는 stars: 42를 받았습니다."
    assert llm.calls == 4  # dispatcher 1 + voter 3
    assert result["iteration"] == 4
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "stargazers_count" not in tool_messages[0].content  # 원본 JSON이 아니라 정리된 텍스트


def test_graph_votes_for_the_candidate_that_matches_tool_facts(fake_llm_factory, mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(200, json_data=[{"path": "main.py", "type": "file", "size": 500}]),
            FakeResponse(200, text="def main(): pass"),
        ]
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_repo_source_sample",
                    "args": {"repo": "octocat/hello-world"},
                    "id": "call_1",
                }
            ],
        ),
        # 세 voter의 독립 시도. voter_1만 실제 도구 결과("def main")를 정확히
        # 반영했고, 나머지 둘은 사실을 언급하지 못했다(환각/회피).
        AIMessage(content="main.py에는 def main(): pass 코드가 있습니다."),
        AIMessage(content="죄송하지만 소스 코드를 확인하지 못했습니다."),
        AIMessage(content="아마 진입점 파일인 것 같습니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="octocat/hello-world 코드 리뷰해줘")],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    # 세 voter 중 사실("def main(): pass")을 실제로 담은 답만 최종 답변으로 채택돼야 한다.
    assert result["messages"][-1].content == "main.py에는 def main(): pass 코드가 있습니다."
    assert len(result["report_drafts"]) == 3


def test_graph_fans_out_both_tools_then_fans_out_three_voters(fake_llm_factory, mock_github_get):
    mock_github_get(
        [
            # repo_overview_node
            FakeResponse(200, json_data={"full_name": "octocat/hello-world", "stargazers_count": 42}),
            FakeResponse(200, text="README"),
            # repo_source_node
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(200, json_data=[{"path": "main.py", "type": "file", "size": 500}]),
            FakeResponse(200, text="def main(): pass"),
        ]
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_repo_overview",
                    "args": {"repo": "octocat/hello-world"},
                    "id": "call_overview",
                },
                {
                    "name": "fetch_repo_source_sample",
                    "args": {"repo": "octocat/hello-world"},
                    "id": "call_source",
                },
            ],
        ),
        AIMessage(content="요약: stars 42. 코드 리뷰: def main(): pass"),
        AIMessage(content="요약: stars 42. 코드 리뷰: def main(): pass"),
        AIMessage(content="요약: stars 42. 코드 리뷰: def main(): pass"),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="octocat/hello-world 요약이랑 코드 리뷰 둘 다 해줘")],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    # dispatcher 1번 + voter 3번 = 정확히 4번. 도구 2개와 voter 3개 모두
    # 순차가 아니라 각자의 라운드에서 병렬로 처리됐다는 증거다.
    assert llm.calls == 4
    assert result["iteration"] == 4
    assert len(result["report_drafts"]) == 3

    tool_messages = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert "octocat/hello-world" in tool_messages["call_overview"]
    assert "def main(): pass" in tool_messages["call_source"]

    final_text = result["messages"][-1].content
    assert "stars" in final_text and "def main" in final_text


def test_graph_skips_voting_when_no_tools_needed(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="바로 답할 수 있어요.")])
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="안녕")], "iteration": 0, "report_drafts": []}
    )

    # dispatcher가 도구를 요청하지 않으면 voter 3개도 투표도 전혀 실행되지
    # 않는다(llm.calls == 1) — 불필요한 LLM 호출을 만들지 않는 게 "효율적"
    # 이라는 dispatcher의 역할과 일치한다.
    assert llm.calls == 1
    assert result["iteration"] == 1
    assert result["messages"][-1].content == "바로 답할 수 있어요."
    assert result.get("report_drafts", []) == []


def test_graph_ignores_tool_calls_returned_by_voter_node(fake_llm_factory, mock_github_get):
    # voter 노드가 (실수로든 뭐든) tool_call이 섞인 응답을 내더라도, voter ->
    # vote_for_best_report -> END는 모두 무조건 엣지라 그래프가 그 tool_call을
    # 절대 실행하지 않는다. 안전장치가 "프롬프트를 잘 따르길 바란다"가 아니라
    # 그래프 구조(고정 엣지 + bind_tools 안 된 llm) 자체임을 보여주는 테스트다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"full_name": "octocat/hello-world"}),
            FakeResponse(200, text="README"),
        ]
    )
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_repo_overview", "args": {"repo": "octocat/hello-world"}, "id": "call_1"}
            ],
        ),
        AIMessage(
            content="정답은 이것입니다.",
            tool_calls=[
                {"name": "fetch_repo_overview", "args": {"repo": "other/repo"}, "id": "call_2"}
            ],
        ),
        AIMessage(content="정답은 이것입니다."),
        AIMessage(content="정답은 이것입니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="octocat/hello-world 리뷰해줘")],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    # voter 단계 이후 추가 라운드가 없으므로 llm은 정확히 4번만 호출된다.
    assert llm.calls == 4
    assert result["iteration"] == 4
    # call_2에 대한 ToolMessage는 존재하지 않는다 — 실행되지 않았다는 뜻.
    tool_call_ids = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_call_ids == {"call_1"}
