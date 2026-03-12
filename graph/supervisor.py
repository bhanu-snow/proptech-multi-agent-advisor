from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from graph.market_researcher import market_researcher  
from graph.valuator import valuator_agent

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    country: str
    next_agent: Literal["market_researcher", "valuator", "end"]

llm = get_llm()

def supervisor(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content.lower()

    # Simple keyword routing (modern: can be replaced with small LLM classifier later)
    if any(word in last_msg for word in ["value", "worth", "valuation", "price estimate", "fair price"]):
        next_agent = "valuator"
    elif any(word in last_msg for word in ["market", "trend", "overview", "summarize", "average"]):
        next_agent = "market_researcher"
    else:
        # Fallback: let LLM decide
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the user query into one category: market_research or valuation."),
            ("human", f"Query: {last_msg}\nRespond only with 'market_research' or 'valuation'"),
        ])
        chain = prompt | llm
        decision = chain.invoke({}).content.strip().lower()
        next_agent = "market_researcher" if "market" in decision else "valuator"

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