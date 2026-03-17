from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph.supervisor import graph
from config import COUNTRY, LLM_PROVIDER, DATA_PATHS
from tools.report_generator import generate_pdf_report

load_dotenv()

st.set_page_config(page_title="PropTech Multi-Agent Advisor", layout="wide")

# Sidebar – country selector & status
with st.sidebar:
    st.title("Controls")
    
    # Country dropdown
    available_countries = list(DATA_PATHS.keys())
    selected_country = st.selectbox(
        "Select Country",
        options=available_countries,
        index=available_countries.index(COUNTRY) if COUNTRY in available_countries else 0,
        key="country_selector"
    )
    
    # LLM status
    st.info(f"LLM Provider: **{LLM_PROVIDER.upper()}**")
    st.info(f"Current Country: **{selected_country}**")
    
    # Clear chat button
    if st.button("Clear Chat & Start New", type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(datetime.now().timestamp())  # new thread
        st.rerun()

st.title("🏠 PropTech Multi-Agent Real Estate Advisor")
st.caption(f"Country: {selected_country} | LLM: {LLM_PROVIDER}")
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about real estate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            # Force current country from sidebar
            current_country = st.session_state.get("country_selector", COUNTRY)
            
            # Debug: show what is actually used
            st.caption(f"DEBUG: Using country = {current_country}")
            st.caption(f"DEBUG: File path = {DATA_PATHS.get(current_country, 'Not found')}")

            thread_id = st.session_state.get("thread_id", "default")
            config = {"configurable": {"thread_id": thread_id}}
            
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "country": current_country,  
            }
            
            result = graph.invoke(initial_state, config=config)
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
                country=current_country,
                market_summary=market_sum,
                valuation_summary=val_sum,
                compliance_summary=comp_sum,
                user_query=prompt
            )
            
            st.download_button(
                label="Download Full Report (PDF)",
                data=pdf_buffer,
                file_name=f"proptech_report_{current_country}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key=f"pdf_{prompt[:20]}"  # avoid duplicate keys
            )
            st.session_state.messages.append({"role": "assistant", "content": final_response})