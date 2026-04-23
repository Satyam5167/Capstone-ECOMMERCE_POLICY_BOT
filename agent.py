# agent.py — E-Commerce FAQ Bot shared agent module
# Import and run: from agent import app, embedder, collection

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from datetime import date, timedelta
import numpy as np

class NumpyCollection:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
    def add(self, documents, embeddings, ids, metadatas):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.np_embeddings = np.array(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.np_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.np_embeddings = self.np_embeddings / norms
        
    def query(self, query_embeddings, n_results=3):
        q_emb = np.array(query_embeddings, dtype=np.float32)
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1e-10
        q_emb = q_emb / q_norms
        similarities = np.dot(q_emb, self.np_embeddings.T)
        out_docs = []
        out_meta = []
        for i in range(len(similarities)):
            sim = similarities[i]
            top_indices = np.argsort(sim)[::-1][:n_results]
            out_docs.append([self.documents[idx] for idx in top_indices])
            out_meta.append([self.metadatas[idx] for idx in top_indices])
        return {"documents": out_docs, "metadatas": out_meta}
    
    def count(self):
        return len(self.documents)

from sentence_transformers import SentenceTransformer
import calendar

# ── Constants ──────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# ── Models ─────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Knowledge Base ─────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Return Policy",
        "text": "We offer a 30-day return policy for most items. Items must be in their original condition, unworn, and with tags attached. Final sale items and intimates cannot be returned. Refunds will be issued to the original form of payment within 5-7 business days after we receive the return."
    },
    {
        "id": "doc_002",
        "topic": "Shipping Options",
        "text": "Standard shipping takes 3-5 business days and is free for orders over $50. Express shipping takes 1-2 business days and costs $15. International shipping takes 7-14 business days depending on the destination. Once your order ships, you will receive a tracking number via email."
    },
    {
        "id": "doc_003",
        "topic": "Order Tracking",
        "text": "You can track your order using the tracking link provided in your shipping confirmation email. Alternatively, you can log into your account on our website and view the status under 'Order History'. If your tracking shows delivered but you haven't received it, please check with neighbors or wait 24 hours before contacting support."
    },
    {
        "id": "doc_004",
        "topic": "Product Warranty",
        "text": "All electronics come with a standard 1-year manufacturer warranty covering defects in materials and workmanship. Accidental damage, such as drops or water damage, is not covered. To file a warranty claim, please contact support with your order number and a photo of the defect."
    },
    {
        "id": "doc_005",
        "topic": "Exchanges",
        "text": "If you need a different size or color, we recommend returning the original item for a refund and placing a new order. We do not offer direct exchanges at this time to ensure the new item is in stock and reaches you as quickly as possible."
    },
    {
        "id": "doc_006",
        "topic": "Damaged or Incorrect Items",
        "text": "If you receive a damaged or incorrect item, please contact our support team within 48 hours of delivery. Provide your order number and photos of the item. We will arrange for a free return shipping label and immediately send out a replacement or issue a full refund."
    },
    {
        "id": "doc_007",
        "topic": "Payment Methods",
        "text": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, and Google Pay. We also offer financing through Klarna for split payments over time. Your payment method will be charged at the time the order is placed."
    },
    {
        "id": "doc_008",
        "topic": "Cancellation Policy",
        "text": "Orders can be cancelled within 1 hour of placement. After this window, the order begins processing in our warehouse and cannot be modified or cancelled. If you miss the cancellation window, you can return the item once you receive it according to our return policy."
    },
    {
        "id": "doc_009",
        "topic": "International Customs",
        "text": "For international orders, customers are responsible for any customs duties, taxes, or import fees imposed by their country's customs department. These fees are not included in our shipping charges and must be paid by the recipient upon delivery."
    },
    {
        "id": "doc_010",
        "topic": "Loyalty Program",
        "text": "Our rewards program allows you to earn 1 point for every $1 spent. Every 100 points equals a $10 discount on a future purchase. Points expire after 12 months of inactivity. Members also receive early access to sales and a special birthday gift."
    }
]

# ── Numpy DB ───────────────────────────────────────────────
collection = NumpyCollection()
texts = [d["text"] for d in DOCUMENTS]
collection.add(
    documents=texts,
    embeddings=embedder.encode(texts).tolist(),
    ids=[d["id"] for d in DOCUMENTS],
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
)


# ── State ──────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


# ── Nodes ──────────────────────────────────────────────────
def memory_node(state):
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}


def router_node(state):
    question = state["question"]
    messages = state.get("messages", [])
    recent = (
        "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1])
        or "none"
    )
    prompt = f"""You are a router for an E-Commerce FAQ Bot used by online shoppers.
Available routes:
- retrieve: search the knowledge base (return policies, shipping info, product catalogue)
- memory_only: answer from conversation history (e.g. 'what did you just say?')
- tool: use the dates tool for questions about today's date, estimated delivery dates, or return windows
Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
    decision = llm.invoke(prompt).content.strip().lower()
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    return {"route": decision}


def retrieval_node(state):
    q_emb = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state):
    return {"retrieved": "", "sources": []}


def tool_node(state):
    today = date.today()
    lines = [f"Today's date: {today.strftime('%B %d, %Y')} ({today.isoformat()})"]
    lines.append("\nUpcoming timelines from today:")
    
    standard_delivery = today + timedelta(days=5)
    express_delivery = today + timedelta(days=2)
    return_window = today + timedelta(days=30)
    
    lines.append(f"  • Estimated Standard Delivery: {standard_delivery.strftime('%B %d, %Y')}")
    lines.append(f"  • Estimated Express Delivery: {express_delivery.strftime('%B %d, %Y')}")
    lines.append(f"  • Return Window Deadline for items bought today: {return_window.strftime('%B %d, %Y')}")
    
    return {"tool_result": "\n".join(lines)}


def answer_node(state):
    question = state["question"]
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TODAY'S DATE & TIMELINES:\n{tool_result}")
    context = "\n\n".join(context_parts)
    if context:
        system_content = f"""You are an E-Commerce FAQ Bot helping online shoppers.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data.

{context}"""
    else:
        system_content = "You are an E-Commerce FAQ Bot. Answer based on the conversation history."
    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Answer using ONLY information explicitly stated in the context above."
    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=question))
    return {"answer": llm.invoke(lc_msgs).content}


def eval_node(state):
    answer = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)
    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}
    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
Context: {context}
Answer: {answer[:300]}"""
    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state):
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


# ── Graph ──────────────────────────────────────────────────
def route_decision(state):
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state):
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


g = StateGraph(CapstoneState)
g.add_node("memory", memory_node)
g.add_node("router", router_node)
g.add_node("retrieve", retrieval_node)
g.add_node("skip", skip_retrieval_node)
g.add_node("tool", tool_node)
g.add_node("answer", answer_node)
g.add_node("eval", eval_node)
g.add_node("save", save_node)
g.set_entry_point("memory")
g.add_edge("memory", "router")
g.add_conditional_edges(
    "router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
)
g.add_edge("retrieve", "answer")
g.add_edge("skip", "answer")
g.add_edge("tool", "answer")
g.add_edge("answer", "eval")
g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
g.add_edge("save", END)

app = g.compile(checkpointer=MemorySaver())
print("agent.py loaded - E-Commerce FAQ Bot ready")
print(f"   Knowledge base: {collection.count()} documents")
