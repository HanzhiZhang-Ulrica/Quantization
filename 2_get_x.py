from pathlib import Path
import json
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================== USER CONFIG (EDIT HERE) ======================
# Local model folder (downloaded by your downloader)
MODEL_DIR = "../model_paths/meta-llama_Llama-3.2-3B-Instruct"

# Device & dtype for the forward pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    # bf16 for Ampere+; otherwise fp16 (you can force float32 if you want)
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    DTYPE = torch.float32

# Prompt text.
TEST_PROMPT = """
[INST]Summarize the following paragraphs:[/INST]

Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had
peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought
Alice "without pictures or conversation?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the
pleasure of making a daisy‐chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit
with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself,
"Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered
at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat‐pocket, and
looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit
with either a waistcoat‐pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and
fortunately was just in time to see it pop down a large rabbit‐hole under the hedge.

Answer:
""".strip()

# Use the tokenizer's chat template (recommended for Instruct models)
USE_CHAT_TEMPLATE = True

# Output directory & format
OUT_DIR = Path("./x_all_layers")
OUTPUT_FORMAT = "pkl"      # Options: "pkl", "csv", "txt"
CSV = OUTPUT_FORMAT == "csv"
DELIM = "," if CSV else " "
FMT = "%.16e"

# Also save RMS-normalized inputs (the ones that multiply Wq/Wk/Wv)?
SAVE_X_NORM = True

# Also save attention_mask and position_ids (once)?
SAVE_MASKS = True

# ====================================================================


@torch.no_grad()
def run_model_and_collect(text: str):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True, device_map=None
    ).to(DEVICE).eval()

    if USE_CHAT_TEMPLATE:
        msgs = [{"role": "user", "content": text}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        batch = tok(rendered, return_tensors="pt")
    else:
        batch = tok(text, return_tensors="pt")

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states length = num_layers + 1
    # hidden_states[i] is the input to layer i (for i=0..num_layers-1)
    hidden_states = out.hidden_states
    num_layers = model.config.num_hidden_layers

    return model, hidden_states, attention_mask, num_layers


def save_array_3d(arr: np.ndarray, path: Path, name: str = None):
    """Save [B, T, D] array in the specified format."""
    if OUTPUT_FORMAT == "pkl":
        data = {
            'array': arr,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'name': name
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Text format
        B, T, D = arr.shape
        flat = arr.reshape(B * T, D)
        with open(path, "w", buffering=1024 * 1024) as f:
            f.write(f"# shape={list(arr.shape)}\n")
            np.savetxt(f, flat, fmt=FMT, delimiter=DELIM)


def save_array_2d(arr: np.ndarray, path: Path, name: str = None):
    """Save [B, T] (e.g., masks/pos ids) array in the specified format."""
    if OUTPUT_FORMAT == "pkl":
        data = {
            'array': arr,
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
            'name': name
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Text format
        with open(path, "w", buffering=1024 * 1024) as f:
            f.write(f"# shape={list(arr.shape)}\n")
            np.savetxt(f, arr, fmt="%.0f", delimiter=DELIM)


@torch.no_grad()
def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    # HF-style: cumsum over mask - 1; masked positions kept at 0
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos.masked_fill_(attention_mask == 0, 0)
    return pos


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, hidden_states, attention_mask, num_layers = run_model_and_collect(TEST_PROMPT)

    # Save attention mask and position ids once (optional)
    if SAVE_MASKS:
        am_np = attention_mask.cpu().numpy()
        pos_ids = build_position_ids(attention_mask).cpu().numpy()
        
        if OUTPUT_FORMAT == "pkl":
            ext = ".pkl"
        else:
            ext = ".csv" if CSV else ".txt"
            
        save_array_2d(am_np, OUT_DIR / f"attention_mask{ext}", "attention_mask")
        save_array_2d(pos_ids, OUT_DIR / f"position_ids{ext}", "position_ids")
        print(f"Saved attention_mask and position_ids → {OUT_DIR}")

    # For each transformer layer i, the input x_i is hidden_states[i]
    # i ranges 0 .. num_layers-1
    for i in range(num_layers):
        # x_i as fp32 for analysis
        x_i = hidden_states[i].cpu().to(torch.float32).numpy()  # [B, T, d_model]
        
        if OUTPUT_FORMAT == "pkl":
            ext = ".pkl"
        else:
            ext = ".csv" if CSV else ".txt"

        # Save raw layer input
        out_x = OUT_DIR / f"x_layer{i}{ext}"
        save_array_3d(x_i, out_x, f"x_layer{i}")

        meta = {
            "layer": i,
            "d_model": int(x_i.shape[-1]),
            "num_heads": int(model.model.layers[i].self_attn.num_heads),
            "num_kv_heads": int(model.model.layers[i].self_attn.num_key_value_heads),
            "head_dim": int(model.model.layers[i].self_attn.head_dim),
            "rmsnorm_eps": float(getattr(model.model.layers[i].input_layernorm, "eps", 1e-5)),
            "rope_theta": None,
        }

        # Save x_norm too (pre-proj tensor actually used with Wq/Wk/Wv)
        if SAVE_X_NORM:
            x_t = hidden_states[i]  # torch [B, T, d_model], model dtype
            x_norm = (
                model.model.layers[i].input_layernorm(x_t)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            out_xn = OUT_DIR / f"xnorm_layer{i}{ext}"
            save_array_3d(x_norm, out_xn, f"xnorm_layer{i}")

        # Try to record RoPE base (theta) if available in this HF version
        try:
            rope = model.model.layers[i].self_attn.rotary_emb
            # Common attribute names across HF versions
            theta = getattr(rope, "base", None)
            if theta is None:
                theta = getattr(rope, "theta", None)
            meta["rope_theta"] = float(theta) if theta is not None else None
        except Exception:
            pass

        # Write tiny per-layer meta JSON
        with open(OUT_DIR / f"meta_layer{i}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Saved x (layer {i}) → {out_x}  | meta_layer{i}.json  "
              f"{'| xnorm saved' if SAVE_X_NORM else ''}")

    print(f"\nDone. Wrote per-layer x (+xnorm/meta) files to: {OUT_DIR.resolve()}")
