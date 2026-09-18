"""Answer generation and judging (shared across all systems).

The SAME LLM and SAME judge are used for every baseline/ablation so that
differences isolate retrieval / memory structure.

Prompts follow the two benchmarks' own specifications:
  * EverMemBench open-ended answers must be SHORT and must not narrate
    reasoning; multiple-choice answers must ALWAYS commit to a letter.
  * GroupMemBench expects a `Final:` line, and abstention questions expect an
    explicit "no information available" refusal.
"""
from .llm import LLMClient

# --- GroupMemBench-style prompts (open-ended + abstention) -------------------
GROUP_AGENT_SYSTEM = (
    "You are a careful QA agent answering questions about a multi-party "
    "workplace group chat. Use only the retrieved conversation passages.\n"
    "Guidelines:\n"
    "- Pay attention to WHO said what, their role, and WHEN.\n"
    "- The question is asked by a specific user; first-person words ('I', 'my') "
    "refer to that asker, and the answer is often in a message the asker wrote.\n"
    "- Speakers in different roles use different words for the same thing "
    "(e.g. an engineer's 'RLHF' is a manager's 'alignment'). Match meaning, not "
    "surface form.\n"
    "- Prefer the most recent decision when passages conflict; an older value "
    "marked [SUPERSEDED by a later update] is no longer current.\n"
    "- Resolve relative dates ('by EOD Friday') against the message timestamp "
    "and answer with an absolute YYYY-MM-DD date.\n"
    "- If the passages genuinely do not contain the answer, say there is no "
    "information available in the conversation.\n"
    "First give a brief reasoning paragraph. Then give the final answer on a "
    "new line in exactly this format:\nFinal: <answer>\n"
    "Keep the Final line short and specific -- just the answer, no explanation."
)
GROUP_JUDGE_SYSTEM = (
    "You are a strict judge evaluating whether an agent's answer matches the gold answer "
    "for a question. Consider paraphrases correct if they have the same meaning as the "
    "gold answer. First provide a brief reasoning paragraph. Then provide the final "
    "judgment on a new line using the format:\nFinal: Correct\nor\nFinal: Incorrect"
)

# --- EverMemBench-style prompts ---------------------------------------------
EVER_AGENT_MC_SYSTEM = (
    "You are a rigorous question-answering assistant. You will be given retrieved "
    "memories from a multi-person group chat and one multiple-choice question with "
    "four options. Choose the single best answer based ONLY on the memories.\n"
    "- Pay attention to who said what and when.\n"
    "- If memories contain contradictory information, prioritize the most recent, "
    "and apply any later override on top of the base rule.\n"
    "- Prefer the option consistent with rules the group actually established, "
    "even when another option sounds more reasonable in general.\n"
    "- Do not prefer an option merely because it is longer or more detailed.\n"
    "- If the memories do not provide enough information to be certain, you MUST "
    "still pick one option. Choose the option that is least inconsistent with "
    "the memories. Never refuse and never say the information is missing.\n"
    "Output ONLY a single uppercase letter: A or B or C or D."
)
EVER_AGENT_OE_SYSTEM = (
    "You are an intelligent memory assistant. Answer the question using ONLY the "
    "retrieved memories from a multi-person group chat.\n"
    "- Pay special attention to timestamps and to who said what.\n"
    "- If memories are contradictory, prioritize the most recent one.\n"
    "- Always convert relative time references into specific dates.\n"
    "- Distinguish a genuine completion ('this task is now complete', "
    "'deliverables archived') from an earlier intention to finish ('I will "
    "finish this today'); only a genuine completion is the end anchor.\n"
    "- For 'how long' questions, identify the start date and the end date, then "
    "state the interval. If the question asks for WORKING days, exclude "
    "Saturdays and Sundays from the count; otherwise count calendar days.\n"
    "- Be concise and specific: answer in under about 10 words when possible.\n"
    "- Do NOT output any reasoning steps. Output ONLY the answer text."
)
EVER_JUDGE_SYSTEM = (
    "You are an expert grader that determines if answers to questions match a gold "
    "standard answer."
)
EVER_JUDGE_USER = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. The "
    "questions concern a multi-person group chat. Be generous: as long as the "
    "generated answer contains the same key information as the gold answer, it is "
    "CORRECT, even if it is longer. For time-related questions, different formats "
    "(e.g. 'May 7th' vs '7 May' vs '2025-05-07') are equivalent, and a +/- 1 day "
    "difference is acceptable. For multiple choice questions where the gold answer "
    "is a letter (A/B/C/D), the generated answer must match exactly.\n"
    "Question: {question}\nGold answer: {golden_answer}\nGenerated answer: {generated_answer}\n"
    "First give a one-sentence explanation, then finish with the label. Do NOT "
    'include both CORRECT and WRONG. Return the label in JSON format with the key '
    '"label": {{"label": "CORRECT"}} or {{"label": "WRONG"}}'
)


class Answerer:
    def __init__(self, model=None, llm=None):
        self.llm = llm or LLMClient(model=model, max_tokens=512)

    def answer_groupmem(self, question, context, asker=None):
        head = f"Question (asked by {asker}): {question}" if asker else \
               f"Question: {question}"
        user = f"{head}\n\nRetrieved passages:\n{context}"
        return self.llm.complete(GROUP_AGENT_SYSTEM, user)

    def answer_evermem(self, question, context, options=None):
        if options:
            opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
            user = (f"[MEMORIES]\n{context}\n\n[QUESTION]\n{question}\n\n"
                    f"[OPTIONS]\n{opt_text}")
            # Budget must leave room for a reasoning model's hidden thinking
            # tokens. At max_tokens=8 a reasoning model spends the whole budget
            # thinking and returns an empty string, which scored every
            # multiple-choice category at exactly 0.0%.
            return self.llm.complete(EVER_AGENT_MC_SYSTEM, user, max_tokens=1024)
        user = f"[MEMORIES]\n{context}\n\n[QUESTION]\n{question}"
        return self.llm.complete(EVER_AGENT_OE_SYSTEM, user, max_tokens=1024)


class Judge:
    def __init__(self, model=None, llm=None):
        self.llm = llm or LLMClient(model=model, max_tokens=128)

    def judge_groupmem(self, question, gold, answer):
        user = (f"Question: {question}\nGold answer: {gold}\nAgent answer: {answer}\n"
                "Judge this.")
        return self.llm.complete(GROUP_JUDGE_SYSTEM, user)

    def judge_evermem(self, question, gold, answer):
        user = EVER_JUDGE_USER.format(
            question=question, golden_answer=gold, generated_answer=answer)
        return self.llm.complete(EVER_JUDGE_SYSTEM, user)


def extract_final(text):
    """Return the text after the last 'Final:' marker (GroupMemBench format)."""
    import re
    matches = list(re.finditer(r"^\s*Final(?:\s+answer)?\s*:\s*(.*)$", text or "",
                               re.IGNORECASE | re.MULTILINE))
    if matches:
        return matches[-1].group(1).strip()
    return (text or "").strip()


def parse_groupmem_verdict(text):
    """Parse the judge verdict.

    Follows GroupMemBench's documented precedence: negatives are tested first,
    because 'not correct' contains the positive trigger 'correct'.
    Returns 1 (correct), 0 (incorrect), or None (unclear -> excluded).
    """
    import re
    matches = list(re.finditer(r"^\s*Final(?:\s+judgment)?\s*:\s*(.*)$", text or "",
                               re.IGNORECASE | re.MULTILINE))
    tail = matches[-1].group(1).lower() if matches else (text or "").lower()
    if any(w in tail for w in ("incorrect", "wrong", "not correct")):
        return 0
    if "correct" in tail:
        return 1
    low = (text or "").lower()
    if any(w in low for w in ("incorrect", "wrong", "not correct")):
        return 0
    if "correct" in low:
        return 1
    return None


def parse_evermem_verdict(text):
    import re
    m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', text or "")
    if m:
        return 1 if m.group(1) == "CORRECT" else 0
    up = (text or "").upper()
    has_c, has_w = "CORRECT" in up, "WRONG" in up
    if has_c and not has_w:
        return 1
    if has_w and not has_c:
        return 0
    return None


def normalize_mc(answer):
    """Extract a bare A/B/C/D letter from a multiple-choice response.

    A reasoning model may return a short paragraph rather than a lone letter,
    so the *last* stated choice is taken (the conclusion), falling back to the
    first letter mentioned. Returns "" when nothing parseable is present, which
    the caller records as a miss rather than a silent pass.
    """
    import re
    if not answer:
        return ""
    text = answer.strip()
    up = text.upper()
    # a lone letter, possibly punctuated: "A", "A.", "(A)"
    m = re.fullmatch(r"[^A-Z0-9]*([ABCD])[^A-Z0-9]*", up)
    if m:
        return m.group(1)
    # an explicit conclusion, e.g. "the answer is C" -- take the last one
    hits = re.findall(
        r"(?:ANSWER|OPTION|CHOICE|SELECT|PICK)\D{0,12}?\b([ABCD])\b", up)
    if hits:
        return hits[-1]
    # otherwise the last standalone letter mentioned
    hits = re.findall(r"\b([ABCD])\b", up)
    return hits[-1] if hits else ""
