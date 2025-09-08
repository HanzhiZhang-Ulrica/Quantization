
# README

This repo has 3 scripts to work with **Llama-3.2 models**:

## 0_download_models.py
Download models from Hugging Face to `../model_paths/`.

- Edit `hf_token`, `BASE_SAVE_DIR`, and `MODELS_TO_DOWNLOAD` at the top.
- Run:
  ```bash
  python 0_download_models.py
  # or with custom path:
  python 0_download_models.py --model-path /path/to/models
  ```

* Example output:

  ```
  ../model_paths/meta-llama_Llama-3.2-1B-Instruct/
    model.safetensors
    tokenizer.json
    config.json
    ...
  ```

---

## 1\_model\_weights.py

Export every tensor from `model.safetensors` into **one CSV/TXT file per weight**.

* Edit `MODEL_FILE`, `OUT_DIR`, `CSV`, and `NAME_FILTER` at the top.
* Run:

  ```bash
  python 1_model_weights.py
  ```
* Output:

  ```
  ./llama32_1b_weights/model.embed_tokens.weight.csv
  ./llama32_1b_weights/model.layers.0.self_attn.q_proj.weight.csv
  ./llama32_1b_weights/model.layers.0.self_attn.k_proj.weight.csv
  ...
  ```

* **Filtering example** (only attention weights):
  ```python
  NAME_FILTER = lambda n: ".self_attn." in n and any(x in n for x in [".q_proj.", ".k_proj.", ".v_proj.", ".o_proj."])
  ```

---

## 2\_get\_x.py

Run a prompt once and save **layer inputs** (`x_layerN.csv`) and **RMSNorm inputs** (`xnorm_layerN.csv`) for all layers, plus small JSON meta.

* Edit `MODEL_DIR`, `TEST_PROMPT`, `OUT_DIR`, `SAVE_X_NORM`, `SAVE_MASKS`.
* Run:

  ```bash
  python 2_get_x.py
  ```
* Output:

  ```
  ./x_all_layers/x_layer0.csv          # Raw input to layer 0
  ./x_all_layers/xnorm_layer0.csv      # RMSNorm(x) for Wq/Wk/Wv
  ./x_all_layers/meta_layer0.json      # Layer metadata (d_model, num_heads, etc.)
  ./x_all_layers/attention_mask.csv    # Attention mask (if SAVE_MASKS=True)
  ./x_all_layers/position_ids.csv      # Position IDs (if SAVE_MASKS=True)
  ...
  ```

---

## Offline math

Later you can load:

* `xnorm_layerN.csv` + weights (`q_proj.weight.csv`, `k_proj.weight.csv`, `v_proj.weight.csv`)
* Do `q = x_norm @ Wq.T`, etc.
* Reshape using info from `meta_layerN.json` (d_model, num_heads, head_dim, etc.)
* Apply RoPE using `position_ids.csv` and `rope_theta` from meta
* Handle GQA using `num_kv_heads` from meta

This gives you Q/K/V in fp32 for your own attention calculations.

---

## Quick start

1. **Download models:**
   ```bash
   python 0_download_models.py
   ```

2. **Extract weights:**
   ```bash
   python 1_model_weights.py
   ```

3. **Generate activations:**
   ```bash
   python 2_get_x.py
   ```

## Pseudocode for Q/K/V computation

```python
# 1) Load x_norm (RMSNorm output) for layer i
X = load_csv("xnorm_layer{i}.csv")      # shape [B*T, d_model]

# 2) Load projection weights
Wq = load_csv("...q_proj.weight.csv")   # shape [H*Dh, d_model]
Wk = load_csv("...k_proj.weight.csv")   # shape [Hkv*Dh, d_model]
Wv = load_csv("...v_proj.weight.csv")   # shape [Hkv*Dh, d_model]

# 3) Load metadata for reshaping
meta = load_json("meta_layer{i}.json")
B, T = X.shape[0] // meta["d_model"], X.shape[0] // meta["d_model"]  # infer from X
H = meta["num_heads"]
Hkv = meta["num_kv_heads"] 
Dh = meta["head_dim"]

# 4) Multiply (note: weight matrix is [out_dim, in_dim], so use transpose)
Q_flat = X @ Wq.T   # [B*T, H*Dh]
K_flat = X @ Wk.T   # [B*T, Hkv*Dh]
V_flat = X @ Wv.T   # [B*T, Hkv*Dh]

# 5) Reshape to 4D with heads
Q = Q_flat.reshape(B, T, H, Dh)
K = K_flat.reshape(B, T, Hkv, Dh)
V = V_flat.reshape(B, T, Hkv, Dh)

# 6) Apply RoPE (optional - for position encoding)
pos_ids = load_csv("position_ids.csv")  # [B, T]
rope_theta = meta["rope_theta"]
# ... apply rotary position embedding to Q, K ...

# 7) Handle GQA (Grouped Query Attention) - repeat K,V for missing heads
if Hkv < H:
    repeat_factor = H // Hkv
    K = K.repeat_interleave(repeat_factor, dim=2)  # [B, T, H, Dh]
    V = V.repeat_interleave(repeat_factor, dim=2)  # [B, T, H, Dh]

# Now you have Q, K, V ready for attention computation!
```
