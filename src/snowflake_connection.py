"""Snowflake connection helper.

The pipeline only runs SELECT queries. This module deliberately does not
execute any DDL, DML, COPY, PUT, or write-back statements.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.config import DEFAULT_DATABASE, DEFAULT_SCHEMA, DEFAULT_WAREHOUSE, ROOT_DIR


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required .env variable: {name}")
    return value


def get_snowflake_connection() -> Any:
    """Create a Snowflake connection from the project .env file."""
    load_dotenv(ROOT_DIR / ".env")

    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is not installed. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    connection_args = {
        "user": _required_env("SF_USER"),
        "password": _required_env("SF_PASSWORD"),
        "account": _required_env("SF_ACCOUNT"),
        "warehouse": os.getenv("SF_WAREHOUSE", DEFAULT_WAREHOUSE),
        "database": os.getenv("SF_DATABASE", DEFAULT_DATABASE),
        "schema": os.getenv("SF_SCHEMA", DEFAULT_SCHEMA),
    }

    role = os.getenv("SF_ROLE")
    if role:
        connection_args["role"] = role

    return snowflake.connector.connect(**connection_args)
