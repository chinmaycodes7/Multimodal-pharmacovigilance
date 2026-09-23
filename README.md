# ADE Extraction, Classification, Localization & Summarization — Experiments

This repo holds a set of experiments I worked on around **adverse drug event (ADE) detection for cancer drugs**, using a mix of LLMs and vision-language models. It's a personal copy of the notebooks I contributed — **not the official/complete repo for the underlying paper**, and not a claim of authorship over the full project. I'm sharing it here as a record of the work.

## What's in here

The experiments fall into four tasks:

- **`entity_extraction/`** — pulling drug names and adverse-event mentions out of text, using both a fine-tuned NER model (BioBERT) and a few instruction-tuned LLMs (Gemma, Llama, Qwen, MedLLaMA) prompted to extract entities directly.
- **`classification/`** — a simpler yes/no task: does a given piece of text describe an adverse drug reaction at all? Tried across Gemma, Qwen, and MedLLaMA (both a `transformers` version and a quantized `llama.cpp`/GGUF version).
- **`localization/`** — given an image (e.g. a skin reaction photo), asking vision-language models to point out or describe *where* the reaction is. Covers Qwen2-VL, MedGemma, LLaVA-Med, PaliGemma, and Med-Flamingo, plus a smaller probe notebook checking whether the models have a left/right spatial bias.
- **`summarization/`** — generating a short summary of a case/report focused on the adverse event, and scoring it with ROUGE.
- **`evaluation/`** — shared scoring utilities: exact/partial-match P/R/F1 for the extraction task, ROUGE scoring, and a CSV-merging helper used between pipeline stages.

If you want a single notebook to skim to get a feel for the general pattern (load model → prompt → parse output → score), `summarization/medllama_summarization.ipynb` is the most fleshed-out one.

## Setup

```bash
pip install -r requirements.txt
```

`med-flamingo` (used by `localization/medflamingo_localization.ipynb`) isn't on PyPI — install it from source:

```bash
git clone https://github.com/mlfoundations/open_flamingo
pip install -e open_flamingo/
```

Several notebooks load gated Hugging Face checkpoints (Llama, Gemma, etc.). Set a token as an environment variable rather than hardcoding one:

```bash
export HF_TOKEN="your_token_here"
```

## Data

No data, images, or model weights are included — just the pipelines. Notebooks expect input under a relative `./data/...` path; point the `INPUT_DIR` / `INPUT_CSV` / `OUTPUT_CSV` variables near the top of each notebook at your own copies.

## Notes on this copy

These notebooks were cleaned up from my original working files before posting here — tokens and personal file paths removed, cell outputs stripped, a couple of dead/duplicate drafts dropped. Details in [`cleaning_log.txt`](./cleaning_log.txt) if you're curious.

## License / reuse

This repo doesn't carry its own license — it's a personal copy of experiments contributed to a larger, multi-author research project, so reuse terms are up to the paper's authors rather than something I can set unilaterally here. Feel free to read through the code, but check with the project/paper authors before reusing it.
