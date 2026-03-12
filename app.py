import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph.supervisor import graph
from config import COUNTRY

load_dotenv()

st.set_page_config(page_title="PropTech Multi-Agent Advisor", layout="wide")
st.title("🏠 PropTech Multi-Agent Real Estate Advisor")
st.caption(f"Country: {COUNTRY} | LLM: {st.session_state.get('llm_provider', 'ollama')} | Data: local Polars")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about real estate (e.g., 'Summarize Dubai market prices')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents thinking..."):
            config = {"configurable": {"thread_id": "1"}}  # For memory
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "country": COUNTRY,
            }
            result = graph.invoke(initial_state, config=config)
            response = result["messages"][-1].content
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})