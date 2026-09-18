"""Per-speaker profile construction (speaker grounding + profile aggregation).

Each profile is structured into query-addressable blocks so that the retrieval
engine can inject only the block relevant to a query (see ``Query-Adaptive
Profile`` in engine.py), instead of a single monolithic text that dilutes
evidence:

  identity        - name, role/title, department          (who this person is)
  expertise       - skills (with proficiency), interests  (what this person knows)
  behavior        - communication style, personality      (how this person writes)
  style_examples  - verbatim sample utterances + surface  (how this person LOOKS
                    markers (emoji, catchphrases)          on the page)
  activity        - groups, topics, message volume        (what this person works on)

``style_examples`` exists because EverMemBench's Style questions turn on exactly
the surface markers (emoji usage, catchphrases, register) that summarisation
pipelines strip as non-semantic noise -- every evaluated memory system failed
that category for this reason. Verbatim samples preserve them.

The ``text`` field remains for backward compatibility and is the concatenation
of the non-empty blocks.
"""
import re
from collections import Counter, defaultdict

from .retrieval import tokenize

_EMOJI_RE = re.compile(r"\[:[a-z_]+:\]|[\U0001F300-\U0001FAFF☀-➿]")


def _skills_text(skills):
    """Render a Skills_List (list of dicts or strings) into a compact string."""
    if not isinstance(skills, list) or not skills:
        return ""
    if isinstance(skills[0], dict):
        parts = []
        for s in skills:
            name = s.get("skill") or s.get("name") or ""
            prof = s.get("proficiency") or s.get("level") or ""
            parts.append(f"{name}({prof})" if prof else str(name))
        return ", ".join(p for p in parts if p.strip("()"))
    return ", ".join(str(s) for s in skills)


def _style_markers(msgs):
    """Surface-level signals that determine 'how this person writes'."""
    out = []
    texts = [m.get("text") or "" for m in msgs]
    if not texts:
        return out

    emojis = Counter()
    for t in texts:
        emojis.update(_EMOJI_RE.findall(t))
    if emojis:
        rate = sum(emojis.values()) / len(texts)
        top = ", ".join(e for e, _ in emojis.most_common(5))
        out.append(f"Emoji usage: {rate:.2f} per message ({top})")
    else:
        out.append("Emoji usage: none")

    lens = [len(t.split()) for t in texts if t]
    if lens:
        out.append(f"Typical message length: {sum(lens) / len(lens):.0f} words")

    q = sum(1 for t in texts if "?" in t) / len(texts)
    ex = sum(1 for t in texts if "!" in t) / len(texts)
    out.append(f"Questioning rate: {q:.0%}; exclamation rate: {ex:.0%}")

    # recurring openers -- catchphrases like "Good question!" are style tells
    openers = Counter()
    for t in texts:
        first = t.strip().split(".")[0].split("!")[0].split("?")[0].strip()
        if 2 <= len(first.split()) <= 5:
            openers[first] += 1
    common = [o for o, c in openers.most_common(3) if c >= 3]
    if common:
        out.append("Recurring openers: " + "; ".join(f'"{o}"' for o in common))
    return out


def build_profiles(messages, profiles_by_name=None, n_examples=3):
    """Return {author: profile_dict} with structured blocks + a joined text.

    For EverMemBench, enriches the gold profiles.json with message-level signals.
    For GroupMemBench, synthesizes a profile from role + message statistics.
    """
    profiles_by_name = profiles_by_name or {}
    by_author = defaultdict(list)
    for m in messages:
        by_author[m.get("author")].append(m)

    # Global token frequency, used to pick each author's distinctive vocabulary.
    total_tokens = Counter()
    for m in messages:
        total_tokens.update(set(tokenize(m.get("text"))))

    out = {}
    for author, msgs in by_author.items():
        role = next((m.get("role") for m in msgs if m.get("role")), None)
        gold = profiles_by_name.get(author, {})
        n = len(msgs)
        groups = Counter(m.get("group") for m in msgs)
        phases = Counter(m.get("phase") for m in msgs if m.get("phase"))
        n_decisions = sum(1 for m in msgs if m.get("is_decision"))

        # Distinctive vocabulary: terms this author uses far more than others.
        author_tokens = Counter()
        for m in msgs:
            author_tokens.update(set(tokenize(m.get("text"))))
        distinctive = [
            t for t, c in author_tokens.most_common(60)
            if total_tokens[t] <= 3 and len(t) > 4
        ][:10]

        # -- identity -----------------------------------------------------
        identity_parts = [f"Speaker: {author}"]
        if role:
            identity_parts.append(f"Role/Title: {role}")
        if gold:
            if gold.get("Title"):
                identity_parts.append(f"Title: {gold['Title']}")
            if gold.get("Dept"):
                identity_parts.append(f"Department: {gold['Dept']}")
            for k in ("Rank", "Level", "Seniority"):
                if gold.get(k):
                    identity_parts.append(f"{k}: {gold[k]}")
        identity = "\n".join(identity_parts)

        # -- expertise ----------------------------------------------------
        expertise_parts = []
        if gold:
            skills = _skills_text(gold.get("Skills_List") or gold.get("Skills"))
            if skills:
                expertise_parts.append("Skills (with proficiency): " + skills)
            interests = gold.get("Interests")
            if isinstance(interests, list) and interests:
                expertise_parts.append("Interests: " +
                                       ", ".join(str(x) for x in interests))
        if distinctive:
            expertise_parts.append("Distinctive vocabulary: " +
                                   ", ".join(distinctive))
        expertise = "\n".join(expertise_parts)

        # -- behavior -----------------------------------------------------
        behavior_parts = []
        if gold:
            comm = (gold.get("Communication_Profile")
                    or gold.get("Communication_Style"))
            if isinstance(comm, dict) and comm:
                behavior_parts.append("Communication style: " + "; ".join(
                    f"{k}={v}" for k, v in comm.items()))
            big5 = gold.get("Big_Five_Profile") or gold.get("Personality")
            if isinstance(big5, dict) and big5:
                behavior_parts.append("Personality (Big Five): " + "; ".join(
                    f"{k}={v}" for k, v in big5.items()))
        behavior_parts.extend(_style_markers(msgs))
        behavior = "\n".join(behavior_parts)

        # -- style examples (verbatim, markers preserved) -------------------
        # Pick medium-length messages: one-word acks carry no style signal.
        ranked = sorted(msgs, key=lambda m: abs(len((m["text"] or "").split()) - 25))
        samples = [m["text"].strip().replace("\n", " ")
                   for m in ranked[:n_examples] if (m["text"] or "").strip()]
        style_examples = ""
        if samples:
            style_examples = ("Verbatim messages written by this person "
                              "(preserve their register, emoji and phrasing):\n"
                              + "\n".join(f'- "{s[:300]}"' for s in samples))

        # -- activity -----------------------------------------------------
        activity_parts = [f"# messages: {n}"]
        if n_decisions:
            activity_parts.append(f"# decision points authored: {n_decisions}")
        if groups:
            activity_parts.append("Groups: " +
                                  ", ".join(str(g) for g, _ in groups.most_common(5)))
        if phases:
            activity_parts.append("Topics: " +
                                  ", ".join(str(p) for p, _ in phases.most_common(5)))
        activity = "\n".join(activity_parts)

        text = "\n".join(filter(None, [identity, expertise, behavior,
                                       style_examples, activity]))

        out[author] = {
            "text": text,
            "identity": identity,
            "expertise": expertise,
            "behavior": behavior,
            "style_examples": style_examples,
            "activity": activity,
            "role": role,
            "gold": gold,
            "n_messages": n,
        }
    return out
