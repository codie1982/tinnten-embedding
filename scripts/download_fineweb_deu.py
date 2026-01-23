#!/usr/bin/env python3
"""Download the DEU train shard of the FineWeb-2 dataset."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copy2

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


def main() -> None:
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN or HUGGINGFACE_TOKEN (https://huggingface.co/settings/tokens) before running this script."
        )

    repo_id = "HuggingFaceFW/fineweb-2"
    filename = "data/deu_Latn/train/000_00000.parquet"
    cache_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=token,
    )

    dest = Path("/home/codie/developer/huggingface/fineweb-2/fineweb-2/data/deu_Latn/train")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(filename).name
    copy2(cache_path, target)
    print(f"Downloaded {filename} → {target}")


if __name__ == "__main__":
    main()
