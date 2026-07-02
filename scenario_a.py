"""Scenario A — API-based annotation (OpenAI).

This file contributes nothing but a backend adapter and a CLI. All annotation
logic is shared with Scenario B in codemix_task.annotators, so the two scenarios
differ only in how a (system, user) prompt pair reaches a model.
"""

import argparse
import os

from codemix_task.annotators import run_samples

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

MODEL = "gpt-4o-mini"


def make_call(model=MODEL):
    """Return a call_fn(system, user) -> raw JSON text backed by the OpenAI API."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(system, user):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return resp.choices[0].message.content

    return call


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scenario A: API-based annotation for sampled code-mixed chat turns."
    )
    parser.add_argument("--csv", default="data/codemix_nura_task.csv")
    parser.add_argument("--out", default="outputs/scenario_a/gpt_4o_mini_samples.jsonl")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=18)
    parser.add_argument("--sample-offset", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output, written = run_samples(
        call_fn=make_call(args.model),
        model=args.model,
        csv_path=args.csv,
        output_path=args.out,
        limit=args.limit,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        sample_offset=args.sample_offset,
    )
    print(f"Saved {written} annotations to {output}")
