"""LangChain chains. Phase 1: a plain chat chain (prompt | llm | parser)."""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.llm import get_llm

SYSTEM_PROMPT = "You are a helpful, concise AI assistant. Answer clearly and say so when you are unsure."

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def to_messages(history: list[dict]):
    """Convert [{role, content}] from the client into LangChain messages."""
    out = []
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        out.append(cls(content=m["content"]))
    return out


def build_chat_chain():
    return chat_prompt | get_llm()


# ---------------- Phase 3: RAG chain ----------------
RAG_SYSTEM = """You answer questions using ONLY the context below, taken from the user's documents.
- Cite sources inline like [1], [2] using the numbers in the context.
- If the answer is not in the context, say you could not find it in the documents. Do not guess.

Context:
{context}"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def build_rag_chain():
    return rag_prompt | get_llm()


def text_of(msg) -> str:
    """Extract plain text from a model message/chunk.
    Newer models (e.g. Gemini 3.x) may return content as a list of blocks instead of a string."""
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                out.append(part.get("text", ""))
        return "".join(out)
    return ""
