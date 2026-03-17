from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from tools.data_fetcher import get_real_estate_data
import polars as pl
from .state import AgentState

llm = get_llm()

def valuator_agent(state: AgentState) -> AgentState:
    country = state["country"]
    df_lazy = get_real_estate_data(state, limit=1000)
    df = df_lazy.collect()  # materialize early once

    row_count = df.shape[0]  # correct way to get row count
    
    summary = f"Valuation insights for {country} ..."

    if row_count == 0 or not df.columns:
        summary = (
            f"No usable data loaded for {country} "
            f"(empty after normalization or file issue). "
            f"Available columns: {df.columns if df.columns else 'none'}"
        )
    elif "price" not in df.columns:
        summary = (
            f"No 'price' column found in {country} data after normalization. "
            f"Available columns: {', '.join(df.columns)}. "
            f"Valuation not possible."
        )
    else:
        # Safe stats – use .mean() with ignore_nulls
        avg_price = df["price"].mean() if "price" in df.columns else None
        median_price = df["price"].median() if "price" in df.columns else None
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

        avg_str = f"{avg_price:,.0f}" if avg_price is not None else "N/A"
        median_str = f"{median_price:,.0f}" if median_price is not None else "N/A" 
        summary = (
            f"Valuation insights for {country} ({row_count} records):\n"
            f"Average price: {avg_str}\n"
            f"Median price: {median_str}\n"
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
        "valuation_summary": summary,
        "data_summary": summary
    }