"""
GitHub 저장소 리뷰 에이전트 - 콘솔 인터페이스.

agent.graph.build_graph()로 조립한 그래프를 실제 ChatAnthropic과 연결해서
터미널에서 대화형으로 사용할 수 있게 한다. 그래프에 체크포인터가 없으므로,
대화 기록(messages)은 이 파일이 파이썬 변수로 들고 있다가 매 턴 그래프에
다시 넘겨준다.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from agent.graph import build_graph

# .env 파일을 읽어서 os.environ에 등록한다. ChatAnthropic이 내부적으로
# os.environ["ANTHROPIC_API_KEY"]를 읽어서 사용한다.
load_dotenv()

MODEL_NAME = "claude-sonnet-5"

# 스터디 모드: voter를 3개가 아니라 1개만 써서 다수결 검증 없이 단일 시도를
# 그대로 채택한다. LLM 호출이 4번(dispatcher+voter 3)에서 2번
# (dispatcher+voter 1)으로 줄어 토큰을 절반 이하로 아낀다 — 무료 티어처럼
# 토큰이 빠듯한 상태에서 대형 저장소를 "가볍게 훑어보며 공부"할 때를 위한
# 설정이다. 정확도(다수결 검증)보다 비용을 우선하는 트레이드오프다.
NORMAL_NUM_VOTERS = 3
STUDY_MODE_NUM_VOTERS = 1


def extract_text(content) -> str:
    """ChatAnthropic 응답의 content를 순수 텍스트 문자열로 변환한다.

    extended thinking이 켜진 모델은 content가 문자열이 아니라
    [{"type": "thinking", ...}, {"type": "text", "text": "..."}] 형태의
    블록 리스트로 온다. 여기서 thinking 블록(추론 과정)은 버리고 text
    블록만 이어붙여야 화면에 실제 답변만 출력된다.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def make_llm() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, max_tokens=4096)


def main() -> None:
    study_mode = input(
        "스터디 모드로 시작할까요? 다수결 검증(voter 3개) 대신 voter 1개만 써서 "
        "LLM 호출을 절반 이하로 줄입니다 — 대형 저장소를 가볍게 훑어보며 "
        "공부할 때 추천합니다. [y/N]: "
    ).strip().lower() in ("y", "yes")
    num_voters = STUDY_MODE_NUM_VOTERS if study_mode else NORMAL_NUM_VOTERS

    graph = build_graph(make_llm(), num_voters=num_voters)
    # report_drafts: 투표형 앙상블(voter_1..voter_N)이 채우는 필드.
    # 명시적으로 빈 리스트를 넣어둬야 operator.add 리듀서가 첫 병렬 쓰기부터
    # 안전하게 누적된다.
    state = {"messages": [], "iteration": 0, "report_drafts": []}

    print(f"\nGitHub 저장소 리뷰 에이전트입니다. ({'스터디' if study_mode else '정식 리뷰'} 모드)")
    print("리뷰할 저장소를 owner/repo 형식으로 알려주세요 (예: 'langchain-ai/langgraph 리뷰해줘').")
    if study_mode:
        print("구조만 가볍게 훑고 싶으면 '구조만 보여줘', 특정 파일을 자세히 보고 싶으면")
        print("'app.py 자세히 보여줘'처럼 요청해보세요.")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        state["messages"].append(("human", user_input))
        state = graph.invoke(state)

        answer = extract_text(state["messages"][-1].content)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
