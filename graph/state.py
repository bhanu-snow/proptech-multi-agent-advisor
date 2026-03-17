from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    country: str
    next_agent: Literal["market_researcher", "valuator", "compliance", "end"]
    market_summary: str = ""
    valuation_summary: str = ""
    compliance_summary: str = ""