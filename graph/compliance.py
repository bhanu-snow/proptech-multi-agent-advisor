import json
from pathlib import Path
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_factory import get_llm
from .state import AgentState

llm = get_llm()

def compliance_agent(state: AgentState) -> AgentState:
    country = state["country"]
    rules_path = Path(f"rules/{country.lower()}_compliance.json")
    summary = f"Compliance overview for {country}:\n{rules_text}"

    if not rules_path.exists():
        summary = f"No compliance rules found for {country}."
    else:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        
        rules_text = "\n".join([
            f"- {r['title']}: {r['description']} (Mandatory: {r['mandatory']})"
            for r in rules.get("key_rules", [])
        ])
        
        summary = f"Compliance overview for {country}:\n{rules_text}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a real-estate compliance expert. Explain rules clearly, flag risks based on user query/context, suggest next steps. Be concise and professional."),
        ("human", "{summary}\nUser query: {query}\nPrevious context: {context}"),
    ])
    
    chain = prompt | llm
    context = state.get("data_summary", "No prior data.")
    response = chain.invoke({
        "summary": summary,
        "query": state["messages"][-1].content,
        "context": context
    })

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "compliance_summary": summary,
        "data_summary": summary
    }