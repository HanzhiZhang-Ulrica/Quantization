from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

# ====================== USER CONFIG (EDIT HERE) ======================
# Path to the single-file checkpoint
MODEL_FILE = Path("../model_paths/meta-llama_Llama-3.2-1B-Instruct/model.safetensors")

# Output directory for weight dumps
OUT_DIR = Path("./llama32_1b_weights")

# Output format: CSV=True (comma), CSV=False -> TXT (space)
CSV = True
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


def main():
    assert MODEL_FILE.exists(), f"Missing {MODEL_FILE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with safe_open(str(MODEL_FILE), framework="pt") as sf:
        for name in sf.keys():
            if not NAME_FILTER(name):
                continue

            t = sf.get_tensor(name)
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(UPCAST_DTYPE)
            arr = t.cpu().numpy()

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


if __name__ == "__main__":
    main()
