"""Download the last RLVR checkpoint of Olmo 3 7B Think into model_cache_test/.

The Think repo's step_XXXX branches are the RLVR training checkpoints
(main is the final release). We take the highest step branch.
"""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "allenai/Olmo-3-7B-Think"
CACHE = ROOT / "model_cache_test"


def hf_token() -> str:
    for line in (ROOT / "API.md").read_text().splitlines():
        if line.startswith("hf read token:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no 'hf read token' line in API.md")


def main():
    token = hf_token()
    branches = HfApi(token=token).list_repo_refs(REPO).branches
    last = max(b.name for b in branches if b.name.startswith("step_"))
    print(f"Last RLVR checkpoint: {last}")
    CACHE.mkdir(exist_ok=True)
    path = snapshot_download(REPO, revision=last, cache_dir=CACHE, token=token)
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
