import ast
import csv
import re


LANG_TAGS = {"EN", "ZH", "MS", "TA", "HK", "SG", "NE", "OTH", "UNK"}
HARM_TYPES = {"hate_speech", "offensive", "neither"}
AGGRESSION = {"genuine_aggression", "mock_aggression", "none"}
TARGETS = {"self", "addressee", "third_party", "group", "none"}
CONFIDENCE = {"certain", "uncertain"}

SG_TERMS = {
    "lah", "leh", "lor", "meh", "sia", "hor", "bah", "mah", "liao", "aiya",
    "paiseh", "walao", "wah", "ah", "sian", "liddat", "siao",
}
OFFENSIVE_TERMS = {
    "shit", "wtf", "fuck", "fucking", "damn", "crap", "useless", "stupid",
    "idiot", "dumb", "bitch", "pissed",
}

def tokenize(message: str) -> list[str]:
    pattern = re.compile(
        r"https?://\S+|"
        r"@\w+|#\w+|"
        r"[A-Za-z]+(?:['’][A-Za-z]+)*"
        r"|[\u4e00-\u9fff]+"
        r"|\d+(?:\.\d+)?"
        r"|[^\sA-Za-z'’\u4e00-\u9fff\d]+"
    )
    return pattern.findall(str(message))


# to get data that has high chance of being offensive
def score_message(message: str) -> int:
    tokens = [token.lower() for token in tokenize(message)]
    text = str(message)
    score = 0
    score += 3 * sum(token in SG_TERMS for token in tokens)
    score += 4 * sum(token in OFFENSIVE_TERMS for token in tokens)
    score += 3 if re.search(r"[\u4e00-\u9fff]", text) else 0
    score += 2 if re.search(r"https?://", text) else 0
    score += 2 if any(ord(char) > 0x2190 for char in text) else 0
    score += 1 if any(token in {"u", "ur", "idk", "alr", "btr", "idw", "tmr"} for token in tokens) else 0
    return score


# get a set of samples from the dataset
def sample_turns(
    csv_path="data/codemix_nura_task.csv",
    limit=8,
    context_window=3,
    sample_mode="sequential",
    max_tokens=None,
    offset=0,
):
    candidates = []
    with open(csv_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        emitted = 0
        seen = 0
        for row_number, row in enumerate(reader):
            messages = ast.literal_eval(row["final_message_list"])
            speakers = ast.literal_eval(row["final_speaker_tracking"])
            relationship = row.get("relationship_type", "")

            for turn_index, message in enumerate(messages):
                text = str(message).strip()
                if not text:
                    continue
                if max_tokens is not None and len(tokenize(text)) > max_tokens:
                    continue

                start = max(0, turn_index - context_window)
                speaker = speakers[turn_index] if turn_index < len(speakers) else None
                item = {
                    "conversation_row": row_number,
                    "conversation_id": row.get("Unnamed: 0", str(row_number)),
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "relationship_type": relationship,
                    "context_turns": messages[start:turn_index],
                    "message": text,
                }

                if sample_mode == "interesting":
                    item["score"] = score_message(text)
                    candidates.append(item)
                    continue

                if seen < offset:
                    seen += 1
                    continue
                yield item
                emitted += 1
                if emitted >= limit:
                    return

    if sample_mode == "interesting":
        candidates.sort(key=lambda item: (-item["score"], item["conversation_row"], item["turn_index"]))
        for item in candidates[offset:offset + limit]:
            item.pop("score", None)
            yield item
