"""
AWS S3 utilities mirroring the Node.js helper functions.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional
from threading import Lock

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .aws_config import get_aws_config


class S3ConnectionError(RuntimeError):
    """Raised when the S3 client cannot be constructed or a call fails."""


_s3_client = None
_lock = Lock()


def get_s3_client(refresh: bool = False):
    """
    Return a cached boto3 S3 client configured from environment variables.
    """
    global _s3_client

    if _s3_client is not None and not refresh:
        return _s3_client

    with _lock:
        if _s3_client is not None and not refresh:
            return _s3_client

        cfg = get_aws_config()
        try:
            client = boto3.client(
                "s3",
                region_name=cfg.region,
                aws_access_key_id=cfg.access_key,
                aws_secret_access_key=cfg.secret_key,
            )
        except (BotoCoreError, NoCredentialsError) as exc:
            raise S3ConnectionError(f"S3 client initialization failed: {exc}") from exc

        _s3_client = client
        return _s3_client


def build_upload_params(key: str, data: Any, content_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepare upload parameters replicating the Node helper's `setParam`.
    """
    cfg = get_aws_config()
    params: Dict[str, Any] = {
        "Bucket": cfg.bucket,
        "Key": key,
        "Body": data,
    }
    if content_type:
        params["ContentType"] = content_type
    return params


def build_download_params(key: str) -> Dict[str, Any]:
    """
    Prepare download parameters replicating `setDownloadParam`.
    """
    cfg = get_aws_config()
    return {
        "Bucket": cfg.bucket,
        "Key": key,
    }


def build_stream_params(key: str) -> Dict[str, Any]:
    """
    Prepare stream parameters replicating `setStreamParam`.
    """
    return build_download_params(key)


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """
    Create a signed URL for the given key.
    """
    client = get_s3_client()
    cfg = get_aws_config()
    try:
        return client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": cfg.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except ClientError as exc:
        raise S3ConnectionError(f"Failed to create presigned URL: {exc}") from exc


def get_file_buffer(key: str) -> bytes:
    """
    Fetch an object from S3 and return its contents as bytes.
    """
    client = get_s3_client()
    params = build_download_params(key)
    try:
        response = client.get_object(**params)
        body = response.get("Body")
        if body is None:
            raise S3ConnectionError("S3 response missing body.")
        if hasattr(body, "read"):
            return body.read()
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        if isinstance(body, io.IOBase):
            return body.read()
        raise S3ConnectionError("Unsupported S3 body type.")
    except ClientError as exc:
        raise S3ConnectionError(f"Failed to download object: {exc}") from exc
