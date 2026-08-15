# BrillaAi Questions Synthesis

Fine-tuning pipeline and dataset for synthetically generating STEM quiz questions in the style of the **National Science and Math Quiz (NSMQ)** — Chemistry, Physics, Mathematics, and Biology — using a LoRA-tuned Llama 3 8B model.

## Overview

This repository contains the two artifacts needed to reproduce and extend the question-synthesis pipeline:

| File | Description |
|---|---|
| `Question_Synthesis_(ACTUAL)_Fine_Tuning_LLama_3_8B_(1).ipynb` | Google Colab notebook that fine-tunes Llama 3 8B (4-bit, via [Unsloth](https://github.com/unslothai/unsloth)) on NSMQ-style question data, then uses the tuned model to generate new questions and save them to CSV/SQLite. |
| `Question Synth Dataset.csv` | 15,852 generated question/answer pairs with subject, difficulty, and formatting metadata. |

The model is prompted as a "highly respected college instructor" setting questions for the NSMQ — a competition where first-year college students solve STEM problems across Chemistry, Physics, Mathematics, and Biology — and generates new question/answer pairs following the same style and structure as real quiz questions.

## Dataset (`Question Synth Dataset.csv`)

- **Rows:** 15,852
- **Columns:**

| Column | Description |
|---|---|
| `has_preamble` | `Yes`/`No` — whether the question includes introductory context/setup text |
| `preamble_text` | The preamble content itself, when present |
| `question` | The generated question text (may include LaTeX-style math, e.g. `$4.0 \mathrm{~ms}^{-1}$`) |
| `answer` | The corresponding answer |
| `subject` | Subject area — `Mathematics`, `Chemistry`, `Physics`, `Biology`, or combinations (e.g. `Chemistry, Physics`) for cross-disciplinary questions |
| `question_type` | Numeric code for the question format (see below) |
| `form` | Reserved metadata field (currently unpopulated — `NY` placeholder) |
| `difficulty` | Reserved metadata field (currently unpopulated — `NY` placeholder) |
| `subject_topic` | Reserved metadata field for sub-topic tagging (currently unpopulated — `NY` placeholder) |

**Subject distribution:** roughly balanced across the four core subjects (~3,700–4,000 questions each), with a smaller number of cross-disciplinary questions (e.g. `Chemistry, Physics`, `Mathematics, Physics`).

**Planned question composition** (per the notebook's generation plan, ~15,600 total):
1. Fundamental questions across all subjects — 9,000
2. Riddles — 2,000
3. True/False questions — 4,000

> Note: `form`, `difficulty`, and `subject_topic` are present in the schema as forward-looking metadata fields but are not yet populated (`NY`) in this dataset version.

## Notebook (`Question_Synthesis_..._Fine_Tuning_LLama_3_8B_(1).ipynb`)

Built on top of the [Unsloth](https://github.com/unslothai/unsloth) Colab fine-tuning template. Runs end-to-end in Google Colab with a GPU runtime and covers:

1. **Environment setup** — detects the Colab GPU and installs Unsloth, `trl`, `peft`, `accelerate`, `bitsandbytes`, and (on Ampere+ GPUs) `flash-attn`/`xformers`.
2. **Model loading** — loads a 4-bit quantized `Llama-3-8B` via `unsloth.FastLanguageModel` (`max_seq_length = 2048`).
3. **LoRA setup** — attaches LoRA adapters (`r=16`, `lora_alpha=16`, targeting the attention/MLP projection layers) for parameter-efficient fine-tuning.
4. **Data prep** — formats training examples with a custom Alpaca-style prompt that casts the model as an NSMQ question-setter, using `has_preamble` / `subject` / `question_type` as structured inputs.
5. **Training** — fine-tunes with `trl.SFTTrainer` (batch size 2, gradient accumulation 4, configurable `max_steps`), with before/after GPU memory reporting.
6. **Inference & generation** — runs the tuned model over a list of `(has_preamble, subject, question_type)` prompts to synthesize new question/answer pairs in bulk.
7. **Post-processing** — regex-based extraction of structured fields from raw model output (`extract_info`), CSV writing (`write_to_csv`), preamble/subject cleanup, and conversion of the resulting CSV into a local SQLite database (`SQLdata.db`).
8. **Model export** — saves LoRA adapters locally or to the Hugging Face Hub, with options to merge to 16-bit/4-bit or export as GGUF (for use with `llama.cpp` or GPT4All).

## Requirements

- Google Colab with a GPU runtime (T4/A100/etc.) — the notebook is designed to run there, including a Google Drive mount step for I/O
- Python packages installed by the notebook itself: `torch`, `unsloth`, `transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `pandas`, `gspread`

## Usage

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run the setup cells to install dependencies and mount Google Drive.
3. Run the model-loading and LoRA cells to prepare `Llama-3-8B` for fine-tuning.
4. Point the data-prep cell at your training data (or use `Question Synth Dataset.csv` as a reference/seed set) and run training.
5. Use the inference cells — edit the `prompts` list (`has_preamble`, `subject`, `question_type`) — to generate new questions in bulk; output is appended to a CSV and optionally converted to SQLite.
6. Optionally export the fine-tuned model (LoRA adapters, merged weights, or GGUF) for downstream use.

## License

No license file is currently included in this repository. All rights are reserved by the author unless a license is added.

## Acknowledgments

- Fine-tuning workflow adapted from the [Unsloth](https://github.com/unslothai/unsloth) Colab notebook template for efficient Llama 3 LoRA fine-tuning.
