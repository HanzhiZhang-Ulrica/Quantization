from pathlib import Path
import json
import numpy as np
import torch
import pickle
from safetensors import safe_open

# ====================== USER CONFIG (EDIT HERE) ======================
# Path to the model directory (will auto-detect single or multi-file)
MODEL_DIR = Path("../model_paths/meta-llama_Llama-3.2-3B-Instruct")

# Output directory for weight dumps
OUT_DIR = Path("./llama32_3b_weights")

# Output format: 'pkl' for pickle, 'csv' for CSV (comma), 'txt' for TXT (space)
OUTPUT_FORMAT = "pkl"  # Options: "pkl", "csv", "txt"
CSV = OUTPUT_FORMAT == "csv"
DELIM = "," if CSV else " "
FMT = "%.16e"  # text float format

# Filter which tensors to export; return True to keep, False to skip
# Examples:
# NAME_FILTER = lambda n: ".self_attn." in n
# NAME_FILTER = lambda n: ".q_proj." in n or ".k_proj." in n or ".v_proj." in n or ".o_proj." in n
NAME_FILTER = lambda n: True

# NumPy cannot consume bf16/fp16 directly; upcast for serialization
UPCAST_DTYPE = torch.float32
# ====================================================================


def get_safetensors_files(model_dir: Path):
    """Get list of safetensors files to process."""
    assert model_dir.exists(), f"Missing model directory {model_dir}"
    
    # Check for single file
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]
    
    # Check for multi-file with index
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Get unique filenames from the weight_map
        filenames = set(index_data["weight_map"].values())
        safetensors_files = [model_dir / fname for fname in filenames]
        
        # Verify all files exist
        for file_path in safetensors_files:
            assert file_path.exists(), f"Missing safetensors file {file_path}"
        
        return sorted(safetensors_files)
    
    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json found in {model_dir}")


def save_tensor(name: str, arr: np.ndarray):
    """Save a single tensor array."""
    if OUTPUT_FORMAT == "pkl":
        # Save as pickle with metadata
        data = {
            'array': arr,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'name': name
        }
        ext = ".pkl"
        out_path = OUT_DIR / f"{name}{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Flatten for text readability
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)

        ext = ".csv" if CSV else ".txt"
        out_path = OUT_DIR / f"{name}{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", buffering=1024 * 1024) as f:
            f.write(f"# shape={list(arr.shape)} dtype={arr.dtype}\n")
            for i in range(flat.shape[0]):
                np.savetxt(f, flat[i:i+1], fmt=FMT, delimiter=DELIM)

    print(f"Saved {name} -> {out_path}  shape={arr.shape} dtype={arr.dtype}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all safetensors files to process
    safetensors_files = get_safetensors_files(MODEL_DIR)
    print(f"Found {len(safetensors_files)} safetensors file(s) to process")
    
    # Process each file
    for file_path in safetensors_files:
        print(f"Processing {file_path.name}...")
        
        with safe_open(str(file_path), framework="pt") as sf:
            for name in sf.keys():
                if not NAME_FILTER(name):
                    continue

                t = sf.get_tensor(name)
                if t.dtype in (torch.bfloat16, torch.float16):
                    t = t.to(UPCAST_DTYPE)
                arr = t.cpu().numpy()

                save_tensor(name, arr)


if __name__ == "__main__":
    main()
