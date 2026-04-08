from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class DeploymentConfig:
    watsonx_apikey: str
    watsonx_url: str
    watsonx_project_id: str
    model_id: str
    thread_id: str
    tickets_file: Path
    use_cos: bool
    cos_endpoint: str | None
    cos_instance_crn: str | None
    cos_bucket_name: str | None
    cos_csv_key: str | None


def load_config(base_dir: Path) -> DeploymentConfig:
    load_dotenv(base_dir.parent / ".env")
    load_dotenv(base_dir / ".env")

    apikey = os.getenv("WATSONX_APIKEY") or os.getenv("WML_APIKEY")
    url = os.getenv("WATSONX_URL") or os.getenv("WML_URL") or "https://us-south.ml.cloud.ibm.com"
    project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")

    if not apikey:
        raise ValueError("Missing WATSONX_APIKEY/WML_APIKEY in environment.")
    if not project_id:
        raise ValueError("Missing WATSONX_PROJECT_ID/PROJECT_ID in environment.")

    model_id = os.getenv("LANGGRAPH_MODEL_ID", "ibm/granite-3-8b-instruct")
    thread_id = os.getenv("LANGGRAPH_THREAD_ID", "thread-1")
    tickets_file = Path(
        os.getenv("LANGGRAPH_TICKETS_FILE", str(base_dir / "data" / "tickets.csv"))
    )

    use_cos = os.getenv("LANGGRAPH_USE_COS", "false").strip().lower() in {"1", "true", "yes", "on"}
    cos_endpoint = os.getenv("COS_ENDPOINT")
    cos_instance_crn = os.getenv("COS_INSTANCE_CRN")
    cos_bucket_name = os.getenv("COS_BUCKET_NAME")
    cos_csv_key = os.getenv("COS_CSV_FILE_NAME", "tickets.csv")

    if use_cos and (not cos_endpoint or not cos_instance_crn or not cos_bucket_name):
        raise ValueError(
            "LANGGRAPH_USE_COS=true requires COS_ENDPOINT, COS_INSTANCE_CRN and COS_BUCKET_NAME."
        )

    return DeploymentConfig(
        watsonx_apikey=apikey,
        watsonx_url=url,
        watsonx_project_id=project_id,
        model_id=model_id,
        thread_id=thread_id,
        tickets_file=tickets_file,
        use_cos=use_cos,
        cos_endpoint=cos_endpoint,
        cos_instance_crn=cos_instance_crn,
        cos_bucket_name=cos_bucket_name,
        cos_csv_key=cos_csv_key,
    )
