from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from tools.data_fetcher import get_real_estate_data

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    country: str
    data_summary: str

llm = get_llm()

def valuator_agent(state: AgentState) -> AgentState:
    df_lazy = get_real_estate_data(limit=1000)
    df = df_lazy.collect()  # execute now

    if "price" not in df.columns or len(df) == 0:
        summary = "No price data available for valuation."
    else:
        avg_price = df["price"].mean()
        median_price = df["price"].median()
        if "location" in df.columns:
            top_areas = df.group_by("location").agg(pl.col("price").mean().alias("avg_price")).sort("avg_price", descending=True).head(3)
            top_str = "\n".join([f"{row['location']}: {row['avg_price']:,.0f}" for row in top_areas.iter_rows(named=True)])
        else:
            top_str = "N/A"

        summary = (
            f"Valuation for {state['country']}:\n"
            f"Average price: {avg_price:,.0f}\n"
            f"Median price: {median_price:,.0f}\n"
            f"Top areas by avg price:\n{top_str}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real-estate valuator. Provide concise, realistic valuation insights. Flag obvious outliers or missing data."),
        ("human", "{summary}\nUser query: {query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"summary": summary, "query": state["messages"][-1].content})

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "data_summary": summary
    }