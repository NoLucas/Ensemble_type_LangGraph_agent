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
    graph = build_graph(make_llm())
    # report_drafts: 3-way 투표형 앙상블(voter_1/2/3)이 채우는 필드.
    # 명시적으로 빈 리스트를 넣어둬야 operator.add 리듀서가 첫 병렬 쓰기부터
    # 안전하게 누적된다.
    state = {"messages": [], "iteration": 0, "report_drafts": []}

    print("GitHub 저장소 리뷰 에이전트입니다.")
    print("리뷰할 저장소를 owner/repo 형식으로 알려주세요 (예: 'langchain-ai/langgraph 리뷰해줘').")
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
