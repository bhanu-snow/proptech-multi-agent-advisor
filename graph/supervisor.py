import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from graph.market_researcher import market_researcher  
from graph.valuator import valuator_agent
from graph.compliance import compliance_agent
from .state import AgentState

llm = get_llm()

def supervisor(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content.strip()
    country = state["country"]

    if len(last_msg.split()) <= 1 and len(last_msg) <= 5:
        return {
            "messages": state["messages"] + [
                AIMessage(content="Hey! Drop your real estate question anytime.")
            ],
            "next_agent": "end"
        }
    
    system_prompt = """
    You are the front-door supervisor for a specialized real-estate AI assistant focused ONLY on {country} markets.

    Your responsibilities:
    1. Determine whether the user's query is related to real estate in {country}.
    2. If YES:
    - Classify into EXACTLY one of:
        - market_research
        - valuation
        - compliance
    3. If NO:
    - Respond with a short, friendly reply

    Strict rules:
    - If ANY real estate intent → on_topic
    - If real estate but NOT in {country} → off_topic
    - If on_topic = true → category MUST NOT be null
    - If on_topic = false → category MUST be null

    Return ONLY JSON:

    {{
    "on_topic": true or false,
    "category": "market_research" or "valuation" or "compliance" or null,
    "reply": "text"
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User message: {last_msg}")
    ])

    chain = prompt | llm

    import re

    def extract_json(text):
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        return match.group(0) if match else None

    fallback = {
        "on_topic": False,
        "category": None,
        "reply": f"Could you rephrase? I handle real estate queries in {country}."
    }

    decision = fallback

    for _ in range(2):
        raw = chain.invoke({
            "last_msg": last_msg,
            "country": country
        }).content.strip()

        json_str = extract_json(raw)
        if not json_str:
            continue

        try:
            decision = json.loads(json_str)
            break
        except:
            continue

    on_topic = decision.get("on_topic", False)
    if isinstance(on_topic, str):
        on_topic = on_topic.lower() == "true"

    if on_topic:
        category = (decision.get("category") or "").lower().replace(" ", "_").strip()

        mapping = {
            "market_research": "market_researcher",
            "valuation": "valuator",
            "compliance": "compliance"
        }

        return {
            "messages": state["messages"],
            "next_agent": mapping.get(category, "end")
        }

        return {
            "messages": state["messages"] + [
                AIMessage(content=decision.get("reply", "Ask me anything about real estate."))
            ],
            "next_agent": "end"
        }

workflow = StateGraph(state_schema=AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("market_researcher", market_researcher)
workflow.add_node("valuator", valuator_agent)
workflow.add_node("compliance", compliance_agent)


workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "market_researcher": "market_researcher",
        "valuator": "valuator",
        "compliance": "compliance",
        "end": END,
    }
)
workflow.add_edge("market_researcher", END)
workflow.add_edge("valuator", END)
workflow.add_edge("compliance", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)