from __future__ import annotations

from pathlib import Path

import ibm_boto3
from ibm_botocore.client import Config
from langchain_ibm import ChatWatsonx
from langgraph.prebuilt import create_react_agent

from .config import DeploymentConfig, load_config
from .tools import TOOLS, configure_tools


def build_react_agent(base_dir: Path):
    config: DeploymentConfig = load_config(base_dir)

    cos_client = None
    if config.use_cos:
        cos_client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=config.watsonx_apikey,
            ibm_service_instance_id=config.cos_instance_crn,
            config=Config(signature_version="oauth"),
            endpoint_url=config.cos_endpoint,
        )

    configure_tools(
        config.tickets_file,
        use_cos=config.use_cos,
        cos_client=cos_client,
        cos_bucket=config.cos_bucket_name,
        cos_key=config.cos_csv_key,
    )

    llm = ChatWatsonx(
        model_id=config.model_id,
        url=config.watsonx_url,
        apikey=config.watsonx_apikey,
        project_id=config.watsonx_project_id,
        params={"temperature": 0, "max_new_tokens": 300},
    )

    system_prompt = (
        "You are an IT support ReAct agent. Use tools when needed. "
        "Before creating a ticket, ensure you have issue details and urgency. "
        "If urgency is missing, ask a follow-up question."
    )

    agent = create_react_agent(model=llm, tools=TOOLS, prompt=system_prompt)
    return agent, config


def run_query(agent, config: DeploymentConfig, user_input: str) -> str:
    result = agent.invoke(
        {"messages": [("user", user_input)]},
        config={"configurable": {"thread_id": config.thread_id}},
    )

    messages = result.get("messages", [])
    if not messages:
        return "No response returned by agent."
    return str(messages[-1].content)
