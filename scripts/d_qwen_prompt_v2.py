"""Stage D-GEC prompt (Qwen2.5-7B-Instruct). FROZEN after the dev-subset test (see 03_configs/2026-11-10_D_plan.yaml addendum).
Three self-written few-shot examples (not from the corpus): minimal grammatical edits, spoken style kept, lower case without
punctuation (the convention of the official DD output that the GEC step receives); the third example is already correct.
build(tok) returns the token ids of the shared prefix (system + examples) and a function giving the per-phrase suffix ids
(user turn + assistant header), so that the prefix KV cache can be reused for every phrase."""
SYS = ("You correct grammatical errors in transcribed spoken English from language learners. Make only the minimal changes "
       "needed to make the utterance grammatical. Keep the speaker's words, word order and spoken style (for example 'you know', "
       "'well', 'so'); do not rephrase, and do not add punctuation or capital letters. If the utterance is already grammatical, "
       "repeat it unchanged. Output only the corrected utterance, in lower case, on one line.")
SHOTS = [
    ("yesterday i go to the shop with my brother and we buy some bread",
     "yesterday i went to the shop with my brother and we bought some bread"),
    ("i think that the people in my town is really friendly you know",
     "i think that the people in my town are really friendly you know"),
    ("well i usually spend my weekends at home with my family",
     "well i usually spend my weekends at home with my family"),
]


def messages():
    m = [{"role": "system", "content": SYS}]
    for x, y in SHOTS:
        m += [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
    return m


def build(tok):
    pre = tok.apply_chat_template(messages(), tokenize=False)

    def full(text):
        return tok.apply_chat_template(messages() + [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    probe = full("x")
    assert probe.startswith(pre), "chat template prefix mismatch"
    hf = tok._tokenizer if hasattr(tok, "_tokenizer") else tok
    pre_ids = hf.encode(pre, add_special_tokens=False)

    def suffix_ids(text):
        f = full(text); assert f.startswith(pre)
        return hf.encode(f[len(pre):], add_special_tokens=False)
    return pre_ids, suffix_ids

VERSION = "v2"
