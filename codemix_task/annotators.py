import json
import re
from pathlib import Path

from codemix_task.common import (
    AGGRESSION,
    CONFIDENCE,
    HARM_TYPES,
    LANG_TAGS,
    TARGETS,
    sample_turns,
    tokenize,
)

LANGID_SYSTEM = """You are a word-level language-identification annotator for Singaporean code-mixed chat. You will be given a message that is split into tokens. 
Tag EACH provided token with exactly one label. Do not merge, split, add, or drop tokens.

TAGS:
- EN  : English, INCLUDING informal spellings/abbreviations (u, dat, tgt, alr, btr,
        bef, j=just, idk, lol). These are English by rule — do NOT tag them UNK.
- ZH  : Mandarin — both Chinese-script tokens AND romanized Mandarin / pinyin
        (e.g. "ni hao", "xie xie", "wo"). Romanized Mandarin is ZH, not UNK/EN.
        EXCEPTION: a romanized item conventionalized as a Singlish borrowing
        (e.g. "jiayou" as encouragement) is SG by function, not ZH.
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

DISAMBIGUATION:
- "ya"/"yeah"/"ye" are informal English (EN), NOT SG, unless used clearly as a sentence-final particle. Default these to EN.
- Tag a token SG only when it functions as a discourse particle in THIS message; if a token is plausibly just informal English, prefer EN.
- Romanized Mandarin (pinyin) is ZH, e.g. "ni hao"/"xie xie"/"wo". Do NOT tag it EN
  just because it is in Latin script. A token that is BOTH romanized Mandarin AND a
  conventional Singlish borrowing ("jiayou") is SG by function.

EXAMPLES (one tag per token, in order):
- tokens ["Aiya","I","paiseh","sia","🤭"]  ->  {"tags":["SG","EN","SG","SG","OTH"]}
- tokens ["ya","lah","u","done"]            ->  {"tags":["EN","SG","EN","EN"]}
- tokens ["xie","xie","u","jiayou"]         ->  {"tags":["ZH","ZH","EN","SG"]}

Return STRICT JSON with one tag per input token, in order:
{"tags": ["TAG1","TAG2", ...]}"""


def langid_user(tokens):
    return (
        "Tag exactly these tokens, preserving order and returning one label per token.\n"
        f"There are {len(tokens)} tokens, so return exactly {len(tokens)} tags.\n"
        f"{json.dumps(tokens, ensure_ascii=False)}"
    )

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

CONTEXT-BLEED WARNING: Laughter (hahaha), emoji, capitalisation, and the 
playfulness of SURROUNDING turns are NOT themselves aggression. Judge the
TARGET message's OWN content. If the target message contains no aggressive
surface form, aggression_function is "none" — no matter how jokey or heated the
context is. A laughing or affectionate message is aggression_function="none".

EXAMPLES:
- target "U btr b ready la 😤" (close friends)
    -> {"harm_type":"neither","aggression_function":"mock_aggression","target":"addressee","confidence":"certain","note":""}
- target "this food taste like shit"
    -> {"harm_type":"offensive","aggression_function":"none","target":"none","confidence":"certain","note":""}
- target "HAHAHAHA stop la"
    -> {"harm_type":"neither","aggression_function":"none","target":"none","confidence":"certain","note":""}
 
Return STRICT JSON:
{"harm_type":"...","aggression_function":"...","target":"...","confidence":"...","note":"..."}"""


def harm_user(target_message, context_turns, relationship):
    ctx = "\n".join(f"- {t}" for t in context_turns) if context_turns else "(none)"
    return (
        f"Relationship between speakers: {relationship}\n\n"
        f"Preceding conversation turns:\n{ctx}\n\n"
        f'TARGET message to annotate:\n"{target_message}"'
    )

def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_langid(tags, tokens):
    issues = []
    if len(tags) != len(tokens):
        issues.append(f"length mismatch: {len(tokens)} tokens vs {len(tags)} tags")
    bad = [t for t in tags if t not in LANG_TAGS]
    if bad:
        issues.append(f"invalid lang tags: {bad}")
    pairs = list(zip(tokens, tags)) if len(tags) == len(tokens) else None
    return {"tokens": tokens, "tags": tags, "pairs": pairs, "issues": issues}


def validate_harm(harm):
    issues = []
    if harm.get("harm_type") not in HARM_TYPES:
        issues.append(f"invalid harm_type: {harm.get('harm_type')}")
    if harm.get("aggression_function") not in AGGRESSION:
        issues.append(f"invalid aggression_function: {harm.get('aggression_function')}")
    if harm.get("target") not in TARGETS:
        issues.append(f"invalid target: {harm.get('target')}")
    if harm.get("confidence") not in CONFIDENCE:
        issues.append(f"invalid confidence: {harm.get('confidence')}")
    if (
        harm.get("harm_type") == "neither"
        and harm.get("aggression_function") == "none"
        and harm.get("target") != "none"
    ):
        issues.append("target should be 'none' when harm_type=neither and aggression=none")
    # Mirror rule: an aggressive ACT is directed at someone, so a present
    # aggression_function must carry a non-none target. Catches the qwen slip
    # of mock_aggression paired with target=none. Restricted to the two valid
    # aggressive values so it never fires on missing/invalid aggression output.
    if (
        harm.get("aggression_function") in {"genuine_aggression", "mock_aggression"}
        and harm.get("target") == "none"
    ):
        issues.append("target should not be 'none' when aggression_function is present")
    harm["issues"] = issues
    return harm

def annotate_item(item, call_fn, model, keep_raw=True):
    message = item["message"]
    tokens = tokenize(message)

    langid_raw = call_fn(LANGID_SYSTEM, langid_user(tokens))
    langid = validate_langid(parse_json(langid_raw).get("tags", []), tokens)

    harm_raw = call_fn(
        HARM_SYSTEM,
        harm_user(message, item["context_turns"], item["relationship_type"]),
    )
    harm = validate_harm(parse_json(harm_raw))

    if keep_raw:
        langid["raw"] = langid_raw
        harm["raw"] = harm_raw

    return {**item, "model": model, "langid": langid, "harm": harm}


def run_samples(
    call_fn,
    model,
    csv_path,
    output_path,
    limit,
    context_window,
    max_tokens,
    sample_offset=0,
    keep_raw=True,
):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = list(sample_turns(
        csv_path=csv_path,
        limit=limit,
        context_window=context_window,
        sample_mode="interesting",
        max_tokens=max_tokens,
        offset=sample_offset,
    ))
    with output.open("w", encoding="utf-8") as f:
        for index, item in enumerate(items, 1):
            record = annotate_item(item, call_fn, model, keep_raw=keep_raw)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Annotated {index}/{len(items)}: {item['message'][:80]}")

    return output, len(items)
