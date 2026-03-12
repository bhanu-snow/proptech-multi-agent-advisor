from langchain_core.language_models import BaseChatModel
from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama
# from langchain_snowflake import ChatSnowflake  # uncomment when you have creds/setup

from config import LLM_PROVIDER, AWS_REGION

def get_llm() -> BaseChatModel:
    if LLM_PROVIDER == "bedrock":
        return ChatBedrockConverse(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=AWS_REGION or "us-east-1",
            temperature=0.0,
        )
    elif LLM_PROVIDER == "snowflake_cortex":
        # Placeholder - real impl needs Snowflake session/creds in .env
        raise NotImplementedError("Snowflake Cortex: Add your session setup here")
        # return ChatSnowflake(model="claude-3-5-sonnet", session=...)
    else:  # ollama default
        return ChatOllama(
            model="llama3.1:8b",
            base_url="http://ollama:11434",
            temperature=0.0,
        )