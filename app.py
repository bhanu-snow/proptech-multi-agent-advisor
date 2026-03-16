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
            
            final_state = result

            # Pull summaries from final state   
            market_sum = final_state.get("market_summary", "No market data processed")
            val_sum    = final_state.get("valuation_summary", "No valuation performed")
            comp_sum   = final_state.get("compliance_summary", "No compliance check performed")
            st.write("Debug summaries:", final_state.get("market_summary"), final_state.get("valuation_summary"), final_state.get("compliance_summary"))
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