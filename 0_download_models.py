import os
import argparse
from huggingface_hub import snapshot_download

# ====================== USER CONFIG (EDIT HERE) ======================
hf_token = "your_token_here"

# Base directory where repos should be saved
BASE_SAVE_DIR = "../model_paths"

# Model repos to download (add/remove as you like)
MODELS_TO_DOWNLOAD = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# If you later want larger models, add them here (commented out by default)
# EXTRA_MODELS = ["lmsys/vicuna-7b-v1.5"]
# ====================================================================


def download_full_repo(repo_id: str, base_save_dir: str, token: str | None):
    """
    Downloads the complete repository (all files) from Hugging Face Hub.
    """
    # Create a folder name by replacing "/" with "_" to avoid file system issues.
    save_dir = os.path.join(base_save_dir, repo_id.replace('/', '_'))
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading full repository for: {repo_id}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        repo_type="model",
        token=token,
        local_dir_use_symlinks=False,  # safer on clusters
    )
    print(f"✅ Repository '{repo_id}' downloaded to: {local_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model repositories to local disk.")
    parser.add_argument("--model-path", type=str, default=BASE_SAVE_DIR,
                        help=f"Base directory (default: {BASE_SAVE_DIR})")
    args = parser.parse_args()

    print(f"Downloading {len(MODELS_TO_DOWNLOAD)} models to: {args.model_path}\n")
    for repo in MODELS_TO_DOWNLOAD:
        download_full_repo(repo, args.model_path, hf_token)
