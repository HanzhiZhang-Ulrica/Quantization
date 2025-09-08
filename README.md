Sure — here’s the simple README wrapped in a Markdown code fence so you can copy it directly:

````markdown
# README

This repo has 3 scripts to work with **Llama-3.2 models**:

## 0_download_models.py
Download models from Hugging Face to `../model_paths/`.

- Edit `hf_token`, `BASE_SAVE_DIR`, and `MODELS_TO_DOWNLOAD` at the top.
- Run:
  ```bash
  python 0_download_models.py
````

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

* Edit `MODEL_FILE` and `OUT_DIR`.
* Run:

  ```bash
  python 1_model_weights.py
  ```
* Output:

  ```
  ./llama32_1b_weights/model.layers.0.self_attn.q_proj.weight.csv
  ...
  ```

---

## 2\_get\_x.py

Run a prompt once and save **layer inputs** (`x_layerN.csv`) and **RMSNorm inputs** (`xnorm_layerN.csv`) for all layers, plus small JSON meta.

* Edit `MODEL_DIR`, `TEST_PROMPT`, `OUT_DIR`.
* Run:

  ```bash
  python 2_get_x.py
  ```
* Output:

  ```
  ./x_all_layers/x_layer0.csv
  ./x_all_layers/xnorm_layer0.csv
  ./x_all_layers/meta_layer0.json
  ...
  ```

---

## Offline math

Later you can load:

* `xnorm_layerN.csv` + weights (`q_proj.weight.csv`, `k_proj.weight.csv`, `v_proj.weight.csv`)
* Do `q = x_norm @ Wq.T`, etc.
* Reshape using info from `meta_layerN.json`.

This gives you Q/K/V in fp32 for your own attention calculations.

```
```
