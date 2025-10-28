"""
Lightweight container for AWS credentials pulled from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AWSConfig:
    access_key: str
    secret_key: str
    bucket: str
    region: str


def get_aws_config() -> AWSConfig:
    """
    Read AWS credentials/settings from environment variables.
    """
    access_key = (os.getenv("AWS_ACCESS_KEY") or "").strip()
    secret_key = (os.getenv("AWS_SECRET_KEY") or "").strip()
    bucket = (os.getenv("AWS_S3_BUCKET") or "").strip()
    region = (os.getenv("AWS_REGION") or "").strip()

    if not all([access_key, secret_key, bucket, region]):
        missing = [
            name
            for name, value in [
                ("AWS_ACCESS_KEY", access_key),
                ("AWS_SECRET_KEY", secret_key),
                ("AWS_S3_BUCKET", bucket),
                ("AWS_REGION", region),
            ]
            if not value
        ]
        raise RuntimeError(f"AWS configuration missing required values: {', '.join(missing)}")

    return AWSConfig(
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=region,
    )
