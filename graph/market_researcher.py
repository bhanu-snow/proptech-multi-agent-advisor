from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from tools.data_fetcher import get_real_estate_data
from .state import AgentState

llm = get_llm()
# Market Researcher Agent node
def market_researcher(state: AgentState) -> AgentState:
    country = state["country"]
    df_lazy = get_real_estate_data(state,limit=200)  # small limit for speed
    df = df_lazy.collect()                     # materialize here

    row_count = df.shape[0]                     # correct way
    summary = f"Loaded {row_count} records for {country}."

    if row_count == 0 or not df.columns:
        summary = f"No data loaded for {country} (empty file or loading issue)."
    else:
        summary = f"Loaded {row_count} records for {country}. "
        if "price" in df.columns:
            summary += f"Avg price: {df['price'].mean():,.0f} | "
        if "location" in df.columns:
            unique_locs = df["location"].unique().to_list()[:5]
            summary += f"Sample locations: {', '.join(unique_locs)}"
        else:
            summary += f"Columns available: {', '.join(df.columns[:5])}..."

    # If query is very short/greeting → short friendly response, no heavy summary
    last_msg = state["messages"][-1].content.strip().lower()
    if len(last_msg) < 15 and any(w in last_msg for w in ["hi", "hello", "hey", "greetings", "sup", "yo"]):
        response_text = "Hey bro! What's up? Ask me anything about real estate in UAE/KSA/India — market overview, valuation, compliance, etc."
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a smart real estate researcher. Summarize the data intelligently based on the user's query."),
            ("human", "{summary}\nUser query: {query}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"summary": summary, "query": last_msg})
        response_text = response.content

    return {
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "market_summary": summary,
        "data_summary": summary
    }
