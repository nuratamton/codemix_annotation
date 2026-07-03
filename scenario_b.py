import argparse
import json
import urllib.error
import urllib.request

from codemix_task.annotators import run_samples

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def make_call(model=MODEL, url=OLLAMA_URL, timeout=180):

    def call(system, user):
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start it with `ollama serve` or the Ollama app, "
                f"and confirm `ollama list` shows {model}."
            ) from exc
        return result["message"]["content"]

    return call


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scenario B: local open-source annotation with Ollama."
    )
    parser.add_argument("--csv", default="data/codemix_nura_task.csv")
    parser.add_argument("--out", default="outputs/scenario_b/qwen2_5_3b_samples.jsonl")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=18)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--disclosure",
        action="store_true",
        help="run the optional disclosure annotation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output, written = run_samples(
        call_fn=make_call(args.model, timeout=args.timeout),
        model=args.model,
        csv_path=args.csv,
        output_path=args.out,
        limit=args.limit,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        sample_offset=args.sample_offset,
        include_disclosure=args.disclosure,
    )
    print(f"Saved {written} annotations to {output}")
