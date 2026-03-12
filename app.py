import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="PropTech Multi-Agent Advisor", layout="wide")
st.title("🏠 PropTech Multi-Agent Real Estate Advisor")
st.caption(f"Country: {st.session_state.get('country', 'UAE')} | LLM: {st.session_state.get('llm_provider', 'ollama')}")

st.success("✅ Foundation ready! Local LangGraph + Polars + Docker running.")
st.info("Next step: We will add the 5 real agents one by one. Reply 'Step 1 done' when this app runs.")

if st.button("Pull Ollama model (first time only)"):
    st.write("Run in terminal: docker exec ollama ollama pull llama3.1:8b")