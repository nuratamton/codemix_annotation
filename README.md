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
  --out outputs/scenario_a/gpt_4o_mini_report_samples.jsonl \
  --limit 10
```

Set `OPENAI_API_KEY` in `.env` or in your shell before running real annotations.

## Scenario B: local model

```bash
source .venv/bin/activate
python scenario_b.py \
  --csv data/codemix_nura_task.csv \
  --out outputs/scenario_b/qwen2_5_3b_report_samples.jsonl \
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

The final Scenario A and Scenario B outputs are:

- `outputs/scenario_a/gpt_4o_mini_report_samples.jsonl`
- `outputs/scenario_b/qwen2_5_3b_report_samples.jsonl`

The JSONL files include the source turn, context turns, model name, language-ID labels, harm labels, validator issues, and raw model responses. Validator issues are preserved in the output instead of silently correcting model output, so failed alignments or invalid labels remain auditable.

 **Notes:**

The sampler is deterministic (score-ranked, ties broken by position), so a
`--sample-offset` identifies the exact turns in each file:

- `*_disclosure_samples.jsonl` - ranked turns 1-10
  (`--limit 10 --sample-offset 0 --disclosure`).
- `*_report_samples.jsonl` - ranked turns 11-20
  (`--limit 10 --sample-offset 10`). These back the report's Section 3.6.
- `scenario_b/qwen2_5_3b_samples_v1.jsonl` / `_v2.jsonl` - ranked turns 1-8
  (`--limit 8`, the old default). The two files contain identical model
  output, v2 is v1 re-validated after the aggression/target consistency rule
  was added to the validator, not a fresh model run.
- `outputs/archive/` - earlier development runs kept for provenance. The
  temperature=0 non-reproducibility discussed in the report (Section 3.7) is
  visible in `archive/scenario_b/samples_v2.jsonl` vs `samples_v3.jsonl`:
  the same 5 turns, identical harm labels, but language tags differ on 4 of 5
  records. (`samples_v4.jsonl` differs from both on harm in one consistent
  direction, which reflects a harm-prompt iteration rather than decoding
  noise.) The prompt iteration that added the context-bleed warning is
  visible in `archive/scenario_a/samples_v1.jsonl` ("HAHAHAHA" labelled
  `mock_aggression`) vs `samples_v2.jsonl` (labelled `none`).

## Notebook

Open `notebooks/eda.ipynb`. Paths inside the notebook are relative to the notebook folder, so the dataset is read from `../data/codemix_nura_task.csv`.

## AI Usage

AI assistance was used during this project to improve sentence clarity in the written reports, help examine annotation outputs and edge cases, and provide guidance while implementing and checking the coding tasks. The prompts used were also generated with AI. The annotation schemes, final decisions, output review, and submitted files were all checked and verified before submission.
