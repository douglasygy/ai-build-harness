from __future__ import annotations

import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse.langchain import CallbackHandler
except ImportError:
    from langfuse.callback import CallbackHandler  # fallback for older langfuse SDK versions


def new_run_id() -> str:
    return str(uuid.uuid4())


def get_callback_handler() -> CallbackHandler:
    return CallbackHandler()


def run_config(run_id: str, tags: Optional[list] = None) -> dict:
    return {
        "callbacks": [get_callback_handler()],
        "metadata": {"langfuse_session_id": run_id},
        "tags": tags or [],
    }
