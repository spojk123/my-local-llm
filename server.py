"""LangServe API server hosting the hybrid personal RAG + general LLM chain."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import tiktoken
import uvicorn
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from pydantic import BaseModel, Field

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
KNOWLEDGE_FILE = Path(__file__).resolve().parent / "knowledge.txt"

_embeddings_cache = None
_vectorstore_cache = {"digest": None, "store": None}


def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def get_vectorstore():
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError("프로젝트 폴더에 knowledge.txt가 없습니다.")
    content = KNOWLEDGE_FILE.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError("knowledge.txt가 비어 있습니다.")

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if _vectorstore_cache["digest"] != digest or _vectorstore_cache["store"] is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            length_function=lambda text: len(tokenizer.encode(text, disallowed_special=())),
        )
        docs = [Document(page_content=content, metadata={"source": "knowledge.txt"})]
        chunks = splitter.split_documents(docs)
        _vectorstore_cache["store"] = FAISS.from_documents(chunks, get_embeddings())
        _vectorstore_cache["digest"] = digest

    return _vectorstore_cache["store"]


def is_personal_question(question: str) -> bool:
    personal_pronoun = re.search(
        r"(?:^|\s)(?:나의|내가|나는|내게|내|저의|제가|저는|제게|제|우리|나에게|저에게|나랑|저랑|나|저)(?=\s|[은는이가의을를와과]|[?!.]|$)",
        question,
    )
    personal_topics = (
        "이름", "성명", "나이", "연세", "생일", "생년월일", "출생", "고향", "출신",
        "사는 곳", "거주지", "주소", "연락처", "전화번호", "이메일", "MBTI", "혈액형",
        "소속", "학교", "대학", "대학교", "학과", "학부", "전공", "부전공",
        "학년", "학번", "반", "직업", "직장", "회사", "부서", "직책", "직급",
        "가족", "가족관계", "어머니", "엄마", "아버지", "아빠", "부모님",
        "형제", "자매", "남매", "오빠", "형", "언니", "누나", "동생",
        "반려동물", "강아지", "고양이",
        "취미", "특기", "관심사", "좋아하는", "싫어하는", "성격", "장점", "단점",
        "목표", "꿈", "장래희망", "경력", "경험", "활동", "프로젝트", "자격증",
        "수상", "포트폴리오", "자기소개", "작성자", "프로필",
    )
    return bool(personal_pronoun) or any(topic in question for topic in personal_topics)


def enforce_korean(answer: str) -> str:
    if not re.search(r"[\u3400-\u9fff]", answer):
        return answer

    fixer = ChatOllama(model=MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=2048)
    try:
        reply = fixer.invoke(
            [
                (
                    "system",
                    "다음 문장에서 중국어 표현만 자연스러운 한국어로 바꾸세요. "
                    "원문의 사실과 의미를 바꾸지 말고 새 정보를 추가하지 마세요. "
                    "오직 한국어로 된 수정 문장만 출력하세요.",
                ),
                ("human", answer),
            ]
        )
        updated = str(reply.content).strip()
        return updated if updated else answer
    except Exception:
        return answer


def format_history(raw_history: list[dict[str, Any]]):
    messages = []
    for item in raw_history[-6:]:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


class ChatInput(BaseModel):
    question: str = Field(..., description="질문을 입력하세요.")
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


def invoke_hybrid_llm(data: Any) -> str:
    if isinstance(data, dict):
        question = data.get("question", "")
        raw_history = data.get("chat_history", [])
    elif isinstance(data, str):
        question = data
        raw_history = []
    else:
        question = getattr(data, "question", "")
        raw_history = getattr(data, "chat_history", [])

    llm = ChatOllama(
        model=MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        num_ctx=4096,
    )
    history_messages = format_history(raw_history)

    if is_personal_question(question):
        vectorstore = get_vectorstore()
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 사용자가 직접 제공한 knowledge.txt를 참고하는 한국어 챗봇입니다.

답변 규칙:
1. 참고 자료에 질문의 답이 있으면 그 사실을 우선하여 정확하게 답하세요.
2. 참고 자료에 적혀 있는 소속, 학교, 가족관계, 이름 등은 사용자가 질문하면 그대로 알려주세요.
   문서에 이미 기록된 사실을 개인정보라는 이유로 숨기거나 없다고 말하지 마세요.
3. 질문에서 '나', '내', '저'는 참고 자료에 작성자의 정보가 있을 때 그 작성자를 의미합니다.
4. 참고 자료에 없는 개인적인 사실은 추측하지 말고 '제공된 자료에서 확인할 수 없습니다'라고 답하세요.
5. 일반 상식 질문은 알고 있는 내용으로 답할 수 있습니다.
6. **답변은 한국어로만** 작성하세요. 중국어 문장이나 중국어 표현을 섞지 마세요.
7. 짧고 명확하게 답하세요. 참고 자료와 질문의 정보가 충돌하면 확인을 요청하세요.

참고 자료:
{context}""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "질문: {question}\n한국어 답변:"),
            ]
        )
        chain = rag_prompt | llm | StrOutputParser()
        raw_answer = chain.invoke(
            {
                "context": context,
                "chat_history": history_messages,
                "question": question,
            }
        )
    else:
        general_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 친절하고 정확한 한국어 AI 도우미입니다. "
                    "일반 상식과 지식 질문에는 학습한 내용을 이용해 직접 답변하세요. "
                    "knowledge.txt에 정보가 없다는 이유로 일반 질문에 대한 답변을 거절하지 마세요. "
                    "사용자가 다른 언어를 명시적으로 요청하지 않았다면 자연스러운 한국어로만 답하세요. "
                    "모르는 최신 사실은 사실인 것처럼 추측하지 말고 확인이 필요하다고 알려주세요.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        chain = general_prompt | llm | StrOutputParser()
        raw_answer = chain.invoke(
            {
                "chat_history": history_messages,
                "question": question,
            }
        )

    return enforce_korean(str(raw_answer).strip())


hybrid_chain = RunnableLambda(invoke_hybrid_llm).with_types(
    input_type=ChatInput,
    output_type=str,
)

app = FastAPI(
    title="나만의 LLM API 서버",
    version="1.0",
    description="knowledge.txt RAG + Ollama qwen2.5:3b LangServe 서버",
)

# /llm 과 /qwen2.5:3b 경로 모두에서 Playground 및 API 호출 가능
add_routes(app, hybrid_chain, path="/llm")
add_routes(app, hybrid_chain, path="/qwen2.5:3b")


@app.on_event("startup")
def preload_resources():
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            json.load(response)
        get_vectorstore()
    except Exception as exc:
        print(f"[경고] 초기 리소스 준비 중 확인 필요: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)