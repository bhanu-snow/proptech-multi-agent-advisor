from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm

from tools.data_fetcher import get_real_estate_data

# State definition (persistent across runs)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    country: str
    data_summary: str

llm = get_llm()

# Market Researcher Agent node
def market_researcher(state: AgentState) -> AgentState:
    df = get_real_estate_data(limit=200)
    summary = f"Loaded {len(df)} records for {state['country']}. "
    summary += f"Avg price: {df['price'].mean():,.0f} | "
    summary += f"Top locations: {', '.join(df['location'].unique()[:3])}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real estate market researcher. Summarize the data intelligently."),
        ("human", "{summary}\nUser query: {query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"summary": summary, "query": state["messages"][-1].content})
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "data_summary": summary
    }
