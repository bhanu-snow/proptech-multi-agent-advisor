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
    df = df_lazy.collect()  # materialize early once

    row_count = df.shape[0]  # correct way to get row count

    if row_count == 0 or not df.columns:
        summary = (
            f"No usable data loaded for {state['country']} "
            f"(empty after normalization or file issue). "
            f"Available columns: {df.columns if df.columns else 'none'}"
        )
    elif "price" not in df.columns:
        summary = (
            f"No 'price' column found in {state['country']} data after normalization. "
            f"Available columns: {', '.join(df.columns)}. "
            f"Valuation not possible."
        )
    else:
        # Safe stats – use .mean() with ignore_nulls
        avg_price = df["price"].mean()
        median_price = df["price"].median()

        top_areas_str = "N/A"
        if "location" in df.columns:
            top_areas = (
                df.group_by("location")
                .agg(pl.col("price").mean().alias("avg_price"))
                .sort("avg_price", descending=True)
                .head(3)
            )
            top_areas_str = "\n".join(
                f"{row['location']}: {row['avg_price']:,.0f}"
                for row in top_areas.iter_rows(named=True)
            )

        summary = (
            f"Valuation insights for {state['country']} ({row_count} records):\n"
            f"Average price: {avg_price:,.0f if avg_price is not None else 'N/A'}\n"
            f"Median price: {median_price:,.0f if median_price is not None else 'N/A'}\n"
            f"Top areas by avg price:\n{top_areas_str}"
        )

    # LLM refinement
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional real-estate valuator. Use the summary to answer accurately. "
                   "If data is missing or unreliable, state it clearly and suggest next steps."),
        ("human", "{summary}\nUser query: {query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"summary": summary, "query": state["messages"][-1].content})

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "data_summary": summary
    }