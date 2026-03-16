from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph.supervisor import graph
from config import COUNTRY
from tools.report_generator import generate_pdf_report

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
            config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "country": COUNTRY,
            }
            result = graph.invoke(initial_state, config=config)
            # Extract last message (final response)
            final_response = result["messages"][-1].content
            st.markdown(final_response)
            
            # Try to collect summaries from state (you may need to expose them better later)
            # For MVP: use last message or hard-code placeholders
            market_sum = "Market summary not captured"   # improve later by returning from agents
            val_sum = "Valuation summary not captured"
            comp_sum = "Compliance summary not captured"
            
            # Show download button
            pdf_buffer = generate_pdf_report(
                country=COUNTRY,
                market_summary=market_sum,
                valuation_summary=val_sum,
                compliance_summary=comp_sum,
                user_query=prompt
            )
            
            st.download_button(
                label="Download Full Report (PDF)",
                data=pdf_buffer,
                file_name=f"proptech_report_{COUNTRY}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key=f"pdf_{prompt[:20]}"  # avoid duplicate keys
            )
            st.session_state.messages.append({"role": "assistant", "content": final_response})