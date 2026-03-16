import json
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from graph.market_researcher import market_researcher  
from graph.valuator import valuator_agent

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    country: str
    next_agent: Literal["market_researcher", "valuator", "end"]

llm = get_llm()

def supervisor(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content.strip()

    # Very short messages → quick bypass
    if len(last_msg) <= 5:
        reply = "Hey! 👋 Drop your real estate question whenever you're ready."
        return {
            "messages": state["messages"] + [AIMessage(content=reply)],
            "next_agent": "end"
        }

    # Let LLM decide topic + reply in one call
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the front-door supervisor for a specialized real-estate AI assistant focused ONLY on India, UAE, and KSA markets.

Your job:
1. Decide if the user query is related to real estate in these countries.
2. If YES → classify into exactly one of: 'market_research' or 'valuation'
3. If NO (off-topic, greeting, chit-chat, unrelated) → write a short, friendly, natural reply.
   Be warm and engaging, match the user's tone, but gently remind the topic if needed.

Response format (JSON only, no extra text):
{{
  "on_topic": true/false,
  "category": "market_research" / "valuation" / null,
  "reply": "your natural response text here (only if on_topic=false)"
}}"""),
        ("human", f"User message: {last_msg}")
    ])

    chain = prompt | llm
    raw_response = chain.invoke({}).content.strip()

    try:
        decision = json.loads(raw_response)
    except Exception as e:
        # Fallback if JSON fails or LLM hallucinates
        decision = {
            "on_topic": False,
            "category": None,
            "reply": "Sorry, could you rephrase that? I'm here for real estate questions about India, UAE, KSA 🏠"
        }

    if decision.get("on_topic", False):
        category = decision.get("category")
        if category == "market_research":
            next_agent = "market_researcher"
        elif category == "valuation":
            next_agent = "valuator"
        else:
            next_agent = "end"  # safety net
    else:
        reply = decision.get("reply", "Hey! What's your real estate question today? 😊")
        return {
            "messages": state["messages"] + [AIMessage(content=reply)],
            "next_agent": "end"
        }

    return {"next_agent": next_agent}



workflow = StateGraph(state_schema=AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("market_researcher", market_researcher)
workflow.add_node("valuator", valuator_agent)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "market_researcher": "market_researcher",
        "valuator": "valuator",
        "end": END,
    }
)
workflow.add_edge("market_researcher", END)
workflow.add_edge("valuator", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)