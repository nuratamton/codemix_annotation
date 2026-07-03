# Code-Mixed Chat Annotation Interview Task

This repository contains the exploratory analysis, annotation schemes, runnable
annotation workflows, and final write-up for the Singaporean code-mixed chat
annotation task. The implementation compares an API-backed model workflow
against a local open-source model workflow on the same score-ranked sample set.

## Project layout

- `data/codemix_nura_task.csv` - input conversation dataset used by the scripts.
- `scenario_a.py` - Scenario A runner using the OpenAI API.
- `scenario_b.py` - Scenario B runner using a local Ollama model.
- `codemix_task/common.py` - shared tokenization, scoring, and sampling logic.
- `codemix_task/annotators.py` - shared prompts, parsing, validation, and JSONL writing.
- `outputs/scenario_a/` - Scenario A JSONL annotation outputs.
- `outputs/scenario_b/` - Scenario B JSONL annotation outputs.
- `notebooks/eda.ipynb` - exploratory notebook used to inspect corpus structure.
- `docs/report.pdf` - final task report.
- `docs/annotation_guidelines.pdf` - human annotation guidelines for the primary and secondary tasks.
- `docs/interview_task.pdf` - original task brief.
- `references/gold.txt` - small hand-written calibration set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scenario B also requires Ollama to be running locally with the selected model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

## Sampling

Both scenario runners use the shared `sample_turns` helper in score-ranked
`interesting` mode. This keeps the API and local-model runs comparable by
prioritising short turns that are more likely to contain code-mixing,
Singlish particles, Mandarin script, emoji, URLs, or potentially harmful
language.

Use `--limit` to choose how many turns to annotate and `--sample-offset` to
continue through the ranked list without overwriting earlier samples.

## Scenario A: API model

```bash
source .venv/bin/activate
python scenario_a.py \
  --csv data/codemix_nura_task.csv \
  --out outputs/scenario_a/gpt_4o_mini_samples.jsonl \
  --limit 10
```

Set `OPENAI_API_KEY` in `.env` or in your shell before running real annotations.

## Scenario B: local model

```bash
source .venv/bin/activate
python scenario_b.py \
  --csv data/codemix_nura_task.csv \
  --out outputs/scenario_b/qwen2_5_3b_samples.jsonl \
  --limit 10
```

## Optional disclosure annotation

Both runners can also produce self-disclosure labels with the same sampled
records:

```bash
python scenario_a.py --disclosure --out outputs/scenario_a/gpt_4o_mini_disclosure_samples.jsonl
python scenario_b.py --disclosure --out outputs/scenario_b/qwen2_5_3b_disclosure_samples.jsonl
```

## Outputs

The JSONL files include the source turn, context turns, model name, language-ID
labels, harm labels, validator issues, and raw model responses. Validator
issues are preserved in the output instead of silently correcting model output,
so failed alignments or invalid labels remain auditable.

## Notebook

Open `notebooks/eda.ipynb`. Paths inside the notebook are relative to the
notebook folder, so the dataset is read from `../data/codemix_nura_task.csv`.
