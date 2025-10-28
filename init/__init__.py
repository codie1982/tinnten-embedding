"""
Helpers for initializing shared infrastructure clients used by the embedding service.

This module mirrors the service connections defined in the Node.js codebase, providing
Python equivalents for MongoDB, AWS S3, Elasticsearch, and RabbitMQ.
"""

from .db import get_mongo_client, get_database
from .aws import (
    get_s3_client,
    build_upload_params,
    build_download_params,
    build_stream_params,
    generate_presigned_url,
    get_file_buffer,
)
from .aws_config import get_aws_config
from .elasticsearch_client import get_elasticsearch_client
from .rabbit_connection import get_rabbit_connection, connect_rabbit_with_retry

__all__ = [
    "get_mongo_client",
    "get_database",
    "get_s3_client",
    "build_upload_params",
    "build_download_params",
    "build_stream_params",
    "generate_presigned_url",
    "get_file_buffer",
    "get_aws_config",
    "get_elasticsearch_client",
    "get_rabbit_connection",
    "connect_rabbit_with_retry",
]
