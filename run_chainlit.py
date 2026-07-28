"""
`python -m chainlit run chainlit_app.py`를 대체하는 실행 스크립트.

chainlit/cli/__init__.py가 무조건 `nest_asyncio.apply()`를 호출하는데,
설치된 anyio(4.14.x)에서 이 패치가 이벤트 루프 감지를 깨서
`anyio.NoEventLoopError`가 나며 정적 파일 요청마다 500이 뜬다
(nest_asyncio 1.6.0이 2년 넘게 업데이트가 없어 최신 anyio/uvicorn과
호환이 깨진 것으로 확인됨). 이 앱은 Jupyter 같은 재진입 이벤트 루프가
필요 없으므로, chainlit이 import하기 전에 nest_asyncio.apply()를
no-op으로 바꿔치기해서 문제를 피한다.
"""

import sys
import types

_fake_nest_asyncio = types.ModuleType("nest_asyncio")
_fake_nest_asyncio.apply = lambda *args, **kwargs: None
sys.modules["nest_asyncio"] = _fake_nest_asyncio

from chainlit.cli import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "run"] + sys.argv[1:]
    cli()
