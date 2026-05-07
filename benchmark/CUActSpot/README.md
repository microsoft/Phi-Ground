# CUActSpot

**CUActSpot** is a benchmark and evaluation harness for *GUI grounding* —
the task of mapping a natural-language instruction to a precise click
coordinate on a screenshot. It evaluates a wide range of open-source and
proprietary multimodal models under a single, unified pipeline so that their
results are directly comparable.

This repository contains the evaluation code only. The benchmark data must
be placed in `data/ActSpot/` (not included).

---

## Highlights

- **Unified pipeline.** Every model (Phi-Ground, OpenCUA, UI-TARS, UI-Venus,
  GUI-Owl, EvoCUA, OS-Atlas, UGround, GTA1, MAI-UI, GPT-5.4, …)
  is wrapped behind the same `BaseModelAdapter` interface, so adding a new
  model only requires registering a small set of components.
- **Composable adapters.** A model spec is built from five composable
  pieces — *prompt builder*, *image preprocessor*, *generation backend*,
  *coordinate parser*, and *coordinate transformer* — that can be swapped
  independently from a JSON config.
- **Multiple inference backends.** Built-in support for `transformers`,
  `vllm` (offline), `vllm` HTTP server / OpenAI-compatible API, and Azure
  OpenAI Responses API (GPT-5.4 with computer-use tools).
- **Best-of-N sampling and parallel evaluation** out of the box.
- **Rich result reports** including per-category (Canvas / GUI / Image /
  Table / Text) accuracy and optional per-sample visualization overlays.

---

## Repository layout

```
CUActSpot/
├── run_eval.py                  # Top-level entry point.
├── eval_utils.py                # Per-sample correctness checker.
├── cuactspot/                   # Core package.
│   ├── adapters/                # BaseModelAdapter and composable adapter.
│   ├── backends/                # Generation backends (vLLM, transformers, Azure, ...).
│   ├── components/              # Prompt builders, image preprocessors,
│   │                            # coordinate parsers, coordinate transforms.
│   ├── models/builtins.py       # ModelSpec preset registry.
│   ├── config.py                # JSON config schema.
│   ├── runner.py                # Parallel evaluation runner.
│   ├── evaluator.py             # Correctness aggregation.
│   ├── dataset.py               # Sample loader.
│   └── cli.py                   # Argparse entry point.
├── configs/                     # One JSON per evaluated model.
├── requirements.txt             # Base + Qwen2-VL / Qwen2.5-VL stack.
├── requirements-qwen3vl.txt     # Extra: Qwen3-VL backbone family (GUI-Owl, EvoCUA).
└── requirements-phi-ground.txt  # Isolated: Phi-Ground family.
```

---

## Installation

We strongly recommend a fresh conda / venv environment per model family.
Three requirements files are provided because the open-source backbones
need different `transformers` / `vllm` versions.

```bash
# 1) Most models (Qwen2.5-VL family + GPT-5.4 + UI-TARS + ...).
pip install -r requirements.txt

# 2) Models with a Qwen3-VL backbone (GUI-Owl-1.5, EvoCUA).
pip install -r requirements.txt -r requirements-qwen3vl.txt

# 3) Phi-Ground family (in a SEPARATE environment).
pip install -r requirements-phi-ground.txt
```

PyTorch, vLLM, and transformers wheels are CUDA-specific — install the
CUDA build that matches your driver before the requirements above if pip
picks the wrong one.

---

## Data setup

Download or assemble the ActSpot data (screenshots + JSON annotations)
into:

```
CUActSpot/data/ActSpot/
```

The `data_dir` field in every config points to this directory. If you keep
the data elsewhere, edit `dataset.data_dir` in the config you run.

---

## How a model is executed

CUActSpot ships with **four** different execution modes. Which one a config
uses depends on the model — there is no single command that works for every
checkpoint. Pick the right column from the table below before running.

| Mode | What it does | When to use | Action required before `run_eval.py` |
| --- | --- | --- | --- |
| **A. Cloud API** | Calls a remote endpoint over HTTP (Azure OpenAI Responses). | GPT-5.4. | Export `CUACTSPOT_AZURE_ENDPOINT` and `CUACTSPOT_AZURE_MI_CLIENT_ID`, or set them in the config. |
| **B. vLLM HTTP server** | Talks to a `vllm serve` process via the OpenAI-compatible REST API. | Most open-source models — gives the best throughput and lets you keep the model warm across many runs. | Launch `vllm serve …` in a separate terminal. The eval process is a thin client. |
| **C. In-process vLLM** | Loads the model with `vllm.LLM(...)` *inside* the evaluation process. | Phi-Ground (custom multimodal protocol). | Just run `run_eval.py` — vLLM is started for you. The first call is slow because the engine warms up. |
| **D. In-process transformers** | Loads the model with HuggingFace `transformers` directly. | Models with custom `modeling_*.py` code, atypical processors, or no vLLM support yet. | Just run `run_eval.py` — `transformers` is invoked inline. Slower than vLLM. |

The rest of this section gives a worked example per mode and lists the
configs that belong to each.

### A. Cloud API — GPT-5.4 (Azure)

```bash
export CUACTSPOT_AZURE_ENDPOINT="https://<your-resource>.openai.azure.com/"
export CUACTSPOT_AZURE_MI_CLIENT_ID="<your-managed-identity-client-id>"

python run_eval.py --config configs/gpt5_4_full.json
```

| Model | HF ID | Config |
| --- | --- | --- |
| GPT-5.4 (best preprocessing setting) | `gpt-5.4` (Azure) | `configs/gpt5_4_full.json` |

### B. vLLM HTTP server (separate `vllm serve` process)

These configs **assume a vLLM server is already running** on
`http://0.0.0.0:8000`. Start it in one terminal, then launch the eval in
another. Example with OpenCUA-7B:

```bash
# Terminal 1 — model server.
vllm serve xlangai/OpenCUA-7B \
    --trust-remote-code \
    --served-model-name opencua-7b \
    --host 0.0.0.0 --port 8000

# Terminal 2 — evaluation.
python run_eval.py --config configs/opencua_7b_full.json \
    --visualize outputs/opencua_7b_full_viz
```

The `--served-model-name` value must match the `model` field in the
backend kwargs of the JSON config. For multi-GPU models (32B / 72B), pass
`--tensor-parallel-size <N>` to `vllm serve`.

| Model | HF ID | Config |
| --- | --- | --- |
| OpenCUA-7B | `xlangai/OpenCUA-7B` | `configs/opencua_7b_full.json` |
| OpenCUA-32B | `xlangai/OpenCUA-32B` | `configs/opencua_32b_full.json` |
| UI-TARS-1.5-7B | `ByteDance-Seed/UI-TARS-1.5-7B` | `configs/ui_tars_1_5_7b_full.json` |
| UI-Venus-Ground-7B | `inclusionAI/UI-Venus-Ground-7B` | `configs/ui_venus_ground_7b_full.json` |
| MAI-UI-8B | `Tongyi-MAI/MAI-UI-8B` | `configs/mai_ui_8b_full.json` |
| MAI-UI-2B | `Tongyi-MAI/MAI-UI-2B` | `configs/mai_ui_2b_full.json` |
| OS-Atlas-Base-7B | `OS-Copilot/OS-Atlas-Base-7B` | `configs/os_atlas_7b_full.json` |
| OS-Atlas-Base-4B | `OS-Copilot/OS-Atlas-Base-4B` | `configs/os_atlas_4b_full.json` |
| UGround-V1-7B | `osunlp/UGround-V1-7B` | `configs/uground_v1_7b_full.json` |
| UGround-V1-2B | `osunlp/UGround-V1-2B` | `configs/uground_v1_2b_full.json` |
| *(template)* | any OpenAI-compatible endpoint | `configs/openai_compatible_template.json` |

### C. In-process vLLM (no server)

The eval process loads vLLM itself, so you only need to run one command.
This mode is currently used by the Phi-Ground configs (custom prompt
protocol). Make sure no other process is using the GPU.

```bash
python run_eval.py --config configs/phi_ground_vllm_full.json \
    --visualize outputs/phi_ground_vllm_full_viz
```

| Model | HF ID | Config |
| --- | --- | --- |
| Phi-Ground-4B-7C | `microsoft/Phi-Ground` | `configs/phi_ground_vllm_full.json` |
| Phi-Ground-4B-16C | *local checkpoint* | `configs/phi_ground_4b_16c_full.json` |

> For `phi_ground_4b_16c_full.json`, replace the placeholder
> `<PATH_OR_HF_ID_TO_PHI_GROUND_4B_16C_CHECKPOINT>` with a real local path
> or HF ID before running.

### D. In-process HuggingFace `transformers`

These models use the HF `transformers` library directly — slower but more
flexible than vLLM. Just run `run_eval.py`; the model is loaded into the
eval process the first time a sample arrives.

```bash
python run_eval.py --config configs/gui_owl_8b_instruct_full.json
```

| Model | HF ID | Config |
| --- | --- | --- |
| UI-Venus-Ground-72B | `inclusionAI/UI-Venus-Ground-72B` | `configs/ui_venus_ground_72b_full.json` |
| GUI-Owl-1.5-8B-Instruct | `mPLUG/GUI-Owl-1.5-8B-Instruct` | `configs/gui_owl_8b_instruct_full.json` |
| GUI-Owl-1.5-8B-Think | `mPLUG/GUI-Owl-1.5-8B-Think` | `configs/gui_owl_8b_think_full.json` |
| EvoCUA-8B | `meituan/EvoCUA-8B-20260105` | `configs/evocua_8b_full.json` |
| EvoCUA-32B | `meituan/EvoCUA-32B-20260105` | `configs/evocua_32b_full.json` |
| InfiGUI-R1-3B | `InfiX-ai/InfiGUI-R1-3B` | `configs/infigui_r1_3b_full.json` |
| InfiGUI-G1-7B | `InfiX-ai/InfiGUI-G1-7B` | `configs/infigui_g1_7b_full.json` |
| SE-GUI-7B | `XinBB/SE-GUI-7B` | `configs/se_gui_7b_full.json` |
| GUI-G2-7B | `inclusionAI/GUI-G2-7B` | `configs/gui_g2_7b_full.json` |

---

## GPT-5.4 / Azure setup

The GPT-5.4 backend authenticates with Azure OpenAI using a Managed
Identity. You can configure it in two ways:

1. **Environment variables (recommended):**
   ```bash
   export CUACTSPOT_AZURE_ENDPOINT="https://<your-resource>.openai.azure.com/"
   export CUACTSPOT_AZURE_MI_CLIENT_ID="<your-managed-identity-client-id>"
   ```
2. **In the config JSON**, override `model.backend.kwargs.azure_endpoint`
   and `model.backend.kwargs.managed_identity_client_id`.

The default config uses the best-performing image preprocessing setting
from our paper: force-resize to 1440×900 plus a screenshot retry loop on
parse failures.

---

## Best-of-N sampling

Any config can be turned into a best-of-N run by setting the
`runtime.best_of` field and (optionally) `runtime.temperature`:

```json
"runtime": {
    "best_of": 8,
    "temperature": 0.7,
    "num_workers": 4
}
```

The runner emits both per-attempt and aggregated metrics in the report.

---

## Reproducing a single model

```bash
python run_eval.py \
    --config configs/gui_owl_8b_instruct_full.json \
    --visualize outputs/gui_owl_8b_instruct_viz
```

Output structure:

```
outputs/<run-name>/
├── report.json          # Aggregate metrics + per-sample results.
└── ...                  # Optional: per-sample visualizations.
```

---

## Adding a new model

1. Implement (or reuse) a generation backend in `cuactspot/backends/`.
2. Register a `ModelSpec` preset in `cuactspot/models/builtins.py` —
   pick a prompt builder, image preprocessor, parser, and transformer.
3. Drop a JSON config in `configs/` that references the preset name and
   overrides any per-run settings.

No core code change is needed beyond steps 1–2.

---

## License & citation

This codebase is released for research purposes. Please cite the
accompanying paper if you use this benchmark in your work. License and
citation entries will be filled in upon release.
