"""Streamlit frontend connected to the local LangServe API server."""

import time
from urllib.request import urlopen

import streamlit as st
from langserve import RemoteRunnable

LANGSERVE_URL = "http://127.0.0.1:8000/llm"


def langserve_is_ready(retries: int = 10, delay: float = 1.0) -> bool:
    for _ in range(retries):
        try:
            with urlopen("http://127.0.0.1:8000/openapi.json", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(delay)
    return False


def main():
    st.set_page_config(page_title="나만의 LLM", page_icon="🤖")

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none !important;
            }
            [data-testid="stSidebarCollapsedControl"] {
                display: none !important;
            }
            [data-testid="stSidebar"] + div {
                margin-left: 0 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("나만의 LLM")

    if not langserve_is_ready():
        st.error(
            f"LangServe 서버({LANGSERVE_URL})에 연결할 수 없습니다. "
            "먼저 `python server.py`가 실행 중인지 확인해 주세요."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 질문해 주세요."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("질문을 입력해 주세요.")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("답변 생성 중..."):
                    chain = RemoteRunnable(LANGSERVE_URL)
                    answer = chain.invoke(
                        {
                            "question": question,
                            "chat_history": st.session_state.messages[:-1],
                        }
                    )
                st.markdown(answer)
            except Exception as exc:
                answer = f"답변 생성 오류: {exc}"
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()