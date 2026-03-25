<h1 align="center">SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2603.23483">
    <img src="https://img.shields.io/badge/arXiv-2603.23483-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/papers/2603.23483">
    <img src="https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg" alt="Hugging Face">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0">
  </a>
</p>


<p align="center">
  <a href="#Highlights">Highlights</a> ·
  <a href="#environment-setup">Environment Setup</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#repository-structure">Repository Structure</a> ·
  <a href="#acknowledgements">Acknowledgements</a> ·
  <a href="#license">License</a> ·
  <a href="#citation">Citation</a>
</p>

SpecEyes is a speculative perception and planning framework for agentic multimodal LLMs. It uses a lightweight vision-language model to quickly assess a visual input and question, then applies answer separability gating to either accept the fast answer or defer to a more powerful large model with tool usage. This approach significantly reduces latency and computation in complex multimodal reasoning while maintaining strong accuracy. This repository includes evaluation code, judge scripts, confidence analysis, and result aggregation tools for SpecEyes.


<a id="Highlights"></a>
## Highlights ✨

<p align="center">
<img src="figures/pipeline.png" width="75%">
</p>

| Direction | Description |
| --- | --- |
| Stateful Bottleneck Analysis | Reveal the sequential tool-use dependency limiting latency and concurrency in agentic MLLMs. |
| Agentic-Level Speculation | Propose speculative reasoning that skips full tool invocation loops for easy queries. |
| Answer Separability Gating | Introduce a new confidence metric based on top-K logit gaps to decide safe bypass. |

## Table of Contents
- [Highlights ✨](#highlights-)
- [Table of Contents](#table-of-contents)
- [1. Environment Setup 🛠️](#1-environment-setup-️)
- [2. Quick Start 🚀](#2-quick-start-)
  - [2.1 Prepare Datasets and Models](#21-prepare-datasets-and-models)
  - [2.2 Run the Main Evaluation](#22-run-the-main-evaluation)
  - [2.3 Start the Judge Model](#23-start-the-judge-model)
  - [2.4 Run the Judge Scripts](#24-run-the-judge-scripts)
  - [3.5 Analyze Small-Model Confidence](#35-analyze-small-model-confidence)
- [3. Repository Structure 🗂️](#3-repository-structure-️)
- [4. Acknowledgements 🙏](#4-acknowledgements-)
- [5. License ⚖️](#5-license-️)
- [6. Citation 📚](#6-citation-)


<a id="environment-setup"></a>
## 1. Environment Setup 🛠️

We recommend `Python 3.11`. Install the PyTorch build matching your CUDA version first, then install the project requirements:

```bash
pip install -r requirements.txt
```

Recommended optional packages:

- `flash-attn`: useful for higher throughput on supported GPUs
- `vllm==0.12.0`: recommended in a separate environment for the judge model service

This repository also relies on a patched image-loading behavior in `qwen-vl-utils`. After installing `qwen-vl-utils`, run:

```bash
python scripts/patch_qwen_vl_utils.py
```

<a id="quick-start"></a>
## 2. Quick Start 🚀

### 2.1 Prepare Datasets and Models

Download the datasets and models into the following directories, or pass explicit paths at runtime:

- [V*](https://huggingface.co/datasets/craigwu/vstar_bench/tree/main): `data/vstar`
- [HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench/tree/main): `data/HR-Bench`
- [POPE](https://huggingface.co/datasets/lmms-lab/POPE/tree/main): `data/POPE`
- [Deepeyes](https://huggingface.co/ChenShawn/DeepEyes-7B): `ChenShawn/DeepEyes-7B`
- [Thyme](https://huggingface.co/Kwai-Keye/Thyme-RL): `Kwai-Keye/Thyme-RL`
- [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): `Qwen/Qwen3-VL-2B-Instruct`
- [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct): `Qwen/Qwen2.5-72B-Instruct`

### 2.2 Run the Main Evaluation

```bash
# Deepeyes baseline
python eval_code_deepeyes/SpecEyes.py --baseline

# Deepeyes with confidence gating
python eval_code_deepeyes/SpecEyes.py --score_threshold 0.98

# Thyme baseline
python eval_code_thyme/SpecEyes.py --baseline

# Thyme with confidence gating
python eval_code_thyme/SpecEyes.py --score_threshold 0.98
```

For the code-reasoning variant, replace `SpecEyes.py` with `SpecReason.py`.

### 2.3 Start the Judge Model

```bash
bash scripts/start_qwen2.5_72b_vllm.sh
```

The default judge endpoint is `http://localhost:23333/v1`. Override it with `--api_url` if needed.

### 2.4 Run the Judge Scripts

```bash
bash scripts/run_judges.sh
```

You can also run them manually:

```bash
python judge_code/judge_vstar.py --input_folder eval_results_qwen3vl-2b-Instruct
python judge_code/judge_hr.py --input_folder eval_results_qwen3vl-2b-Instruct
python judge_code/judge_pope.py --input_folder eval_results_qwen3vl-2b-Instruct
```

### 3.5 Analyze Small-Model Confidence

```bash
# Run batched small-model inference
python scripts/small_model_batch_inference.py

# Judge the generated outputs
python judge_code/judge_vstar.py --input_folder eval_results_qwen3vl-2b-Instruct
python judge_code/judge_hr.py --input_folder eval_results_qwen3vl-2b-Instruct

# Analyze judge results
python scripts/analyze_small_confidence.py --input_folder judge_results_qwen3vl-2b-Instruct
python scripts/analyze_small_conf_percentage.py --input_folder judge_results_qwen3vl-2b-Instruct
```

<a id="repository-structure"></a>
## 3. Repository Structure 🗂️

```text
SpecEyes/
├── data/
│   ├── vstar/
│   ├── HR-Bench/
│   └── POPE/
├── eval_code_deepeyes/
├── eval_code_thyme/
├── judge_code/
├── scripts/
├── vis/
├── eval_results_deepeyes/
├── eval_results_thyme/
└── ...
```

Core directories:

| Path | Description |
| --- | --- |
| `eval_code_deepeyes/` | `SpecEyes` and `SpecReason` evaluation code built on Deepeyes |
| `eval_code_thyme/` | `SpecEyes` and `SpecReason` evaluation code built on Thyme |
| `judge_code/` | Judge scripts using a vLLM OpenAI-compatible endpoint |
| `scripts/small_model_batch_inference.py` | Batched small-model inference and confidence signal export |
| `scripts/gather_result.py` | Aggregation of speedup, and accuracy results |
| `scripts/analyze_small_confidence.py` | Confidence-distribution and performance analysis |
| `vis/` | Plotting and visualization utilities used in the paper |

Additional notes:

- `eval_code_thyme/sandbox.py` is a localized sandbox copy used by the Thyme evaluation pipeline
- Temporary processed images are written to `eval_code_thyme/temp_processed_images/`
- Result folders and cache directories are intentionally excluded through `.gitignore`

<a id="acknowledgements"></a>
## 4. Acknowledgements 🙏

This repository benefits from code references from the [DeepEyes](https://github.com/Visual-Agent/DeepEyes) repository. We sincerely thank the authors and maintainers for their open-source contributions, which helped inform parts of our implementation and experimentation workflow.

<a id="license"></a>
## 5. License ⚖️

This repository is released under `Apache-2.0`. See `LICENSE` for the full license text.

The repository also includes notes about third-party code and patches, including:

- the upstream source attribution for `eval_code_thyme/sandbox.py`
- the patching behavior for `qwen-vl-utils`

See `THIRD_PARTY_NOTICES.md` for the relevant attribution and redistribution notes. If you redistribute or modify those third-party-related components, you should also follow the corresponding upstream license requirements.

<a id="citation"></a>
## 6. Citation 📚

If you use this repository, please cite the corresponding paper:

```bibtex
@article{huang2026,
  title={SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning},
  author={Huang, Haoyu and Huang, Jinfa and Wan, Zhongwei and Zheng, Xiawu and Ji, Rongrong and Luo, Jiebo},
  journal={arXiv preprint arXiv:2603.23483},
  year={2026}
}
```
