import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"

def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LANG_TAGS = {"EN", "ZH", "MS", "TA", "HK", "SG", "NE", "OTH", "UNK"}
HARM_TYPES = {"hate_speech", "offensive", "neither"}
AGGRESSION = {"genuine_aggression", "mock_aggression", "none"}
TARGETS = {"self", "addressee", "third_party", "group", "none"}
CONFIDENCE = {"certain", "uncertain"}

def tokenize(message: str) -> list[str]:
    pattern = re.compile(
        r"https?://\S+|"             # URLs stay intact
        r"@\w+|#\w+|"                # handles/hashtags
        r"[A-Za-z]+(?:['’][A-Za-z]+)*"  # latin word-runs, I'm/I’m, HAHAHA
        r"|[\u4e00-\u9fff]+"         # CJK runs (睡觉)
        r"|\d+(?:\.\d+)?"            # numbers
        r"|[^\sA-Za-z'’\u4e00-\u9fff\d]+"  # emoji/punctuation/symbol runs
    )
    return pattern.findall(str(message))

LANGID_SYSTEM = """You are a word-level language-identification annotator for Singaporean code-mixed chat. You will be given a message that is split into tokens. 
Tag EACH provided token with exactly one label. Do not merge, split, add, or drop tokens.

TAGS:
- EN  : English, INCLUDING informal spellings/abbreviations (u, dat, tgt, alr, btr,
        bef, j=just, idk, lol). These are English by rule — do NOT tag them UNK.
- ZH  : Mandarin / Chinese-script tokens.
- MS  : Malay content words.
- TA  : Tamil content words.
- HK  : Hokkien content words.
- SG  : Singlish discourse particles, tagged by FUNCTION not etymology
        (lah, leh, lor, meh, sia, hor, bah, mah, liao, aiya, paiseh, walao, wah, ah).
        NOTE: bare "ah" is SG when it marks a pause/turn ("wait ah"), but it can be an
        English interjection in other contexts — judge from context.
- NE  : Named entities / proper nouns (people, places, brands).
- OTH : Emoji, URLs, numbers, punctuation, non-lexical laughter (hahaha).
- UNK : Genuinely unidentifiable only.

Return STRICT JSON with one tag per input token, in order:
{"tags": ["TAG1","TAG2", ...]}"""

def langid_user_prompt(tokens):
    return (
        "Tag exactly these tokens, preserving order and returning one label per token:\n"
        f"{json.dumps(tokens, ensure_ascii=False)}"
    )

def annotate_langid(message):
    tokens = tokenize(message)
    resp = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": LANGID_SYSTEM},
            {"role": "user", "content": langid_user_prompt(tokens)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)
    tags = parsed.get("tags", [])
 
    issues = []
    if len(tags) != len(tokens):
        issues.append(f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags")
    bad = [t for t in tags if t not in LANG_TAGS]
    if bad:
        issues.append(f"invalid tags: {bad}")
 
    paired = list(zip(tokens, tags)) if len(tags) == len(tokens) else None
    return {"tokens": tokens, "tags": tags, "pairs": paired, "issues": issues}
 


HARM_SYSTEM = """You are a harmful-content annotator for Singaporean code-mixed chat.
Judge a TARGET message using the surrounding conversation and relationship context.
 
AXIS 1 — harm_type (judges FORM/CONTENT):
- "hate_speech": attacks/demeans based on a PROTECTED attribute (race, religion,
   gender, sexuality, nationality, disability).
- "offensive": profane/vulgar/insulting but NOT targeting a protected attribute.
- "neither": no profanity, no attack.
 
AXIS 2 — aggression_function (judges FUNCTION, independent of Axis 1):
- "genuine_aggression": hostile form AND hostile intent — a real attack/threat/anger.
- "mock_aggression": hostile FORM functioning as teasing/banter/bonding (affiliative).
   Cues: laughing emoji, close-friend relationship, reciprocal teasing, hyperbole.
- "none": no aggressive surface form.
 
GOVERNING PRINCIPLE: Axis 1 = form/content; Axis 2 = function. Same words can be
neither+mock ("U btr b" between friends) or offensive+none ("taste like shit" about food).
An angry/crude WORD is not an aggressive ACT — describing one's own feeling ("i was
pissed") is none, not aggression.
 
target: one of self | addressee | third_party | group | none.
  RULE: target is the target OF harm/aggression. If harm_type=="neither" AND
  aggression_function=="none", target MUST be "none".
confidence: "certain" or "uncertain". note: short reason if uncertain, else "".
 
Return STRICT JSON:
{"harm_type":"...","aggression_function":"...","target":"...","confidence":"...","note":"..."}"""

def harm_user_prompt(target_message, context_turns, relationship):
    ctx = "\n".join(f"- {t}" for t in context_turns) if context_turns else "(none)"
    return (
        f"Relationship between speakers: {relationship}\n\n"
        f"Preceding conversation turns:\n{ctx}\n\n"
        f'TARGET message to annotate:\n"{target_message}"'
    )


def annotate_harm(target_message, context_turns, relationship):
    resp = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": HARM_SYSTEM},
            {"role": "user", "content": harm_user_prompt(target_message, context_turns, relationship)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)

    issues = []
    if parsed.get("harm_type") not in HARM_TYPES:
        issues.append(f"invalid harm_type: {parsed.get('harm_type')}")
    if parsed.get("aggression_function") not in AGGRESSION:
        issues.append(f"invalid aggression_function: {parsed.get('aggression_function')}")
    if parsed.get("target") not in TARGETS:
        issues.append(f"invalid target: {parsed.get('target')}")
    if parsed.get("confidence") not in CONFIDENCE:
        issues.append(f"invalid confidence: {parsed.get('confidence')}")
    if (parsed.get("harm_type") == "neither"
            and parsed.get("aggression_function") == "none"
            and parsed.get("target") != "none"):
        issues.append("target should be 'none' when harm_type=neither and aggression=none")
 
    parsed["issues"] = issues
    return parsed


def sample_turns(csv_path="codemix_nura_task.csv", limit=8, context_window=3):
    """Yield a deterministic sample of message turns from the provided dataset."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        emitted = 0
        for row_number, row in enumerate(reader):
            messages = ast.literal_eval(row["final_message_list"])
            speakers = ast.literal_eval(row["final_speaker_tracking"])
            relationship = row.get("relationship_type", "")

            for turn_index, message in enumerate(messages):
                text = str(message).strip()
                if not text:
                    continue

                start = max(0, turn_index - context_window)
                speaker = speakers[turn_index] if turn_index < len(speakers) else None
                yield {
                    "conversation_row": row_number,
                    "conversation_id": row.get("Unnamed: 0", str(row_number)),
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "relationship_type": relationship,
                    "context_turns": messages[start:turn_index],
                    "message": text,
                }

                emitted += 1
                if emitted >= limit:
                    return


def annotate_samples(
    csv_path="codemix_nura_task.csv",
    output_path="outputs/scenario_a_annotations.jsonl",
    limit=10,
    context_window=3,
):
    """Annotate sampled turns and save one JSON record per line."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as f:
        for item in sample_turns(csv_path, limit=limit, context_window=context_window):
            langid = annotate_langid(item["message"])
            harm = annotate_harm(
                item["message"],
                item["context_turns"],
                item["relationship_type"],
            )
            record = {
                **item,
                "model": MODEL,
                "langid": langid,
                "harm": harm,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"Annotated {written}/{limit}: {item['message'][:80]}")

    return output, written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scenario A: API-based annotation for sampled code-mixed chat turns."
    )
    parser.add_argument("--csv", default="data/codemix_nura_task.csv")
    parser.add_argument("--out", default="outputs/scenario_a_annotations.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--demo", action="store_true", help="Run only the two hard-coded demo examples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY set, showing tokenization and prompt examples")
        print("Tokenization 1:", tokenize("Aiya I paiseh sia"))
        print("Tokenization 2:", tokenize("睡觉time alr"))
        print("LANGID SYSTEM PROMPT", LANGID_SYSTEM)
        print("HARM SYSTEM PROMPT", HARM_SYSTEM)

    elif args.demo:
        print("LangID:", annotate_langid("Aiya I paiseh sia"))
        print("Harm:", annotate_harm(
            "you useless",
            ["wah you really cannot run sia", "HAHAHA"],
            "Close - Close friend (trust 4/4)",
        ))
    else:
        output, written = annotate_samples(
            csv_path=args.csv,
            output_path=args.out,
            limit=args.limit,
            context_window=args.context_window,
        )
        print(f"Saved {written} annotations to {output}")
