"""Publish this folder to the Hugging Face Hub as a dataset.

THE OPERATOR LOGS IN FIRST. This script never asks for a token and never prints
one: it uses the login `hf auth login` saved (or HF_TOKEN). It creates the
dataset PRIVATE unless told otherwise, so what lands can be read on the Hub
before anyone else can see it; `--public` flips an existing dataset to public.

    python diditwork/hf_publish.py --repo craigm26/did-it-work            # create or update, private
    python diditwork/hf_publish.py --repo craigm26/did-it-work --public   # make it public

Before uploading it writes MANIFEST.json: the SHA-256 and size of every file it
uploads, so a copy can be checked against the Hub byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc"}


def files() -> list[Path]:
    out = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        if SKIP_PARTS & set(path.relative_to(HERE).parts) or path.suffix in SKIP_SUFFIXES:
            continue
        out.append(path)
    return out


def write_manifest() -> Path:
    entries = {}
    for path in files():
        data = path.read_bytes()
        entries[path.relative_to(HERE).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "package": "did-it-work",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "github.com/craigm26/duckbench, folder diditwork/",
        "note": "sha256 and byte size of every file in this package except MANIFEST.json itself.",
        "fileCount": len(entries),
        "totalBytes": sum(e["bytes"] for e in entries.values()),
        "files": entries,
    }
    path = HERE / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="owner/name of the dataset, e.g. craigm26/did-it-work")
    p.add_argument("--public", action="store_true", help="make the dataset public (after you have read it)")
    p.add_argument("--message", default="did-it-work: spec, harness, vectors and results")
    args = p.parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("needs huggingface_hub: pip install -U huggingface_hub", file=sys.stderr)
        return 2
    api = HfApi()
    who = api.whoami()          # fails loudly if nobody is logged in
    print(f"logged in as {who.get('name')}")
    if args.public:
        api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=False)
        print(f"https://huggingface.co/datasets/{args.repo} is now public")
        return 0
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
    manifest = write_manifest()
    print(f"wrote {manifest.name} ({json.loads(manifest.read_text())['fileCount']} files)")
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(HERE),
                      commit_message=args.message,
                      ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**"])
    print(f"uploaded: https://huggingface.co/datasets/{args.repo} (private until --public)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
