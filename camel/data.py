"""Unified loaders for GroupMemBench and EverMemBench.

Both benchmarks are converted to a common internal message schema:

    Message = dict with keys:
      id       (str)   unique id
      text     (str)   message body
      author   (str)   speaker id / name
      role     (str|None) role label
      ts       (datetime) timestamp
      group    (str)   channel / group
      phase    (str|None) phase / topic
      reply_to (str|None) parent message id
      is_decision (bool)
      is_noise    (bool)
      meta     (dict)  benchmark-specific extras
"""
import json
import re
from datetime import datetime
from pathlib import Path

from . import config

# Fields the engine reads, with safe defaults. A new benchmark's loader only has
# to supply `id`, `text`, `author` and `ts`; normalise() fills in the rest so a
# dataset without threading, phases or decision flags plugs in unchanged.
_MESSAGE_DEFAULTS = {
    "text": "", "author": None, "role": None, "ts": None, "group": None,
    "phase": None, "reply_to": None, "is_decision": False, "is_noise": False,
}


def normalize(messages):
    """Fill in missing schema fields so any loader's output is engine-ready."""
    for m in messages:
        for k, v in _MESSAGE_DEFAULTS.items():
            if k not in m or m[k] is None and v is not False:
                m.setdefault(k, v)
        m.setdefault("meta", {})
        if m.get("is_decision") is None:
            m["is_decision"] = False
        if m.get("is_noise") is None:
            m["is_noise"] = False
    return messages


def _parse_ts(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19] if len(s) >= 19 else s, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# GroupMemBench
# ---------------------------------------------------------------------------
def load_groupmem_domain(domain):
    """Return (messages, channel_of_message) for one domain.

    messages: list of message dicts in chronological order (across all channels).
    channel_of_message: msg_id -> channel name.
    """
    path = config.GROUP_DATA / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"
    data = json.loads(path.read_text())
    messages = []
    channel_of = {}
    for channel, msgs in data.items():
        for m in msgs:
            msg = {
                "id": m.get("msg_node"),
                "text": m.get("content", ""),
                "author": m.get("author"),
                "role": m.get("role"),
                "ts": _parse_ts(m.get("timestamp")),
                "group": channel,
                "phase": m.get("phase_name"),
                "reply_to": m.get("reply_to"),
                "is_decision": bool(m.get("is_decision_point")),
                "is_noise": bool(m.get("is_noise")),
                "meta": {
                    "topic": m.get("topic"),
                    "decision_type": m.get("decision_type"),
                    "decision_label": m.get("decision_label"),
                    "decision_change": m.get("decision_change_metadata"),
                    "phase_order": m.get("phase_order"),
                },
            }
            messages.append(msg)
            channel_of[msg["id"]] = channel
    messages.sort(key=lambda x: (x["ts"] or datetime.min, x["id"]))
    return messages, channel_of


def load_groupmem_questions(domain, qtype=None):
    """Return list of question dicts {id, question, answer, asking_user_id, qtype}."""
    base = config.GROUP_QUESTIONS / domain
    out = []
    qtypes = [qtype] if qtype else config.GROUP_QTYPES
    for qt in qtypes:
        f = base / f"{qt}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            q["qtype"] = qt
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# EverMemBench
# ---------------------------------------------------------------------------
def load_evermem_topic(topic):
    """Return (messages, profiles_by_name) for one topic.

    messages: list of message dicts.  group = 'Group 1/2/3', phase = topic_id,
              author = speaker name, meta['date'], meta['message_index'].
    """
    dlg_path = config.EVER_DATA / topic / "dialogue.json"
    data = json.loads(dlg_path.read_text())
    messages = []
    for entry in data:
        date = entry["date"]
        for gkey, msgs in entry["dialogues"].items():
            if not msgs:
                continue
            for m in msgs:
                msg = {
                    "id": f"{entry['topic_id']}|{date}|{gkey}|{m.get('message_index')}",
                    "text": m.get("dialogue", ""),
                    "author": m.get("speaker"),
                    "role": None,  # filled from profiles below
                    "ts": _parse_ts(m.get("time") or f"{date} 00:00:00"),
                    "group": gkey,
                    "phase": entry.get("topic_id"),
                    "reply_to": None,
                    "is_decision": False,
                    "is_noise": False,
                    "meta": {
                        "date": date,
                        "message_index": m.get("message_index"),
                        "task_ids": m.get("task_ids"),
                    },
                }
                messages.append(msg)
    messages.sort(key=lambda x: (x["ts"] or datetime.min, x["id"]))
    profiles = load_evermem_profiles()
    _attach_roles(messages, profiles)
    return messages, profiles


def load_evermem_questions(topic):
    """Return list of question dicts for one topic.

    Keys: id, question (Q), answer (A), reference (R), options, category, dimension.
    """
    qa_path = config.EVER_DATA / topic / f"qa_{topic}.json"
    data = json.loads(qa_path.read_text())
    out = []
    for q in data:
        qid = q["id"]
        prefix = "_".join(qid.split("_")[:2])
        dim, cat = config.EVER_CATEGORIES.get(prefix, ("Unknown", prefix))
        out.append({
            "id": qid,
            "question": q["Q"],
            "answer": q["A"],
            "reference": q.get("R"),
            "options": q.get("options"),
            "category": cat,
            "dimension": dim,
        })
    return out


def load_evermem_profiles():
    """Return {name: profile_dict} from profiles.json."""
    path = config.EVER_DATA / "profiles.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    profs = {}
    for p in data:
        name = p.get("Name")
        if name:
            profs[name] = p
    return profs


def _attach_roles(messages, profiles):
    for m in messages:
        p = profiles.get(m["author"])
        if p:
            m["role"] = p.get("Title") or p.get("Dept")


# ---------------------------------------------------------------------------
# SocialMemBench  (Owolabi, arXiv:2605.17789; CC BY 4.0)
# ---------------------------------------------------------------------------
# Text-only multi-party social group chat: 43 networks, 430 personas, 348
# sessions, 7,355 turns, 1,031 QA. Released as four parquet configs on the Hub
# (`conversations`, `qa`, `networks`, `personas`) joined on `network_id`.
#
# One NETWORK maps to one corpus, the way a GroupMemBench domain or an
# EverMemBench topic does: memory persists within a network and resets between
# networks, which matches the benchmark's own evaluation protocol.
SOCIAL_QTYPES = {
    "Q1": "single-contact-recall", "Q2": "group-decision",
    "Q3": "multi-contact-aggregation", "Q4": "attribution-probe",
    "Q5": "theory-of-mind", "Q6": "norm-vs-individual",
    "Q7": "relational-edge", "Q8": "temporal-shift",
    "Q9": "departed-member",
}


def _social_frames():
    """Load the four parquet tables, from a local dir or the Hub."""
    import pandas as pd
    base = getattr(config, "SOCIAL_DATA", None)
    frames = {}
    for name in ("conversations", "qa", "networks", "personas"):
        path = (base / f"{name}.parquet") if base else None
        if path is not None and path.exists():
            frames[name] = pd.read_parquet(path)
        else:
            frames[name] = pd.read_parquet(
                "hf://datasets/anon4data/socialmembench/" f"{name}.parquet")
    return frames


def load_socialmem_network(network_id, _cache={}):
    """Return (messages, profiles_by_name) for one social network."""
    if "f" not in _cache:
        _cache["f"] = _social_frames()
    f = _cache["f"]
    conv = f["conversations"]
    conv = conv[conv["network_id"] == network_id]

    messages = []
    for r in conv.itertuples(index=False):
        messages.append({
            "id": r.turn_id,
            "text": r.message or "",
            "author": r.speaker_display_name,
            "role": None,                    # filled from personas below
            "ts": _parse_ts(str(r.timestamp)),
            # A session is this benchmark's conversational unit; there is one
            # chat per network, so `group` is the network and `phase` the
            # session -- mirroring how phases work in the other two benchmarks.
            "group": r.network_id,
            "phase": r.session_id,
            "reply_to": (r.reply_to_turn_id or None),
            "is_decision": False,
            "is_noise": False,
            "meta": {"session_index": int(r.session_index),
                     "session_topic": r.session_topic,
                     "message_index": int(r.message_index),
                     "persona_id": r.speaker_persona_id},
        })
    messages.sort(key=lambda x: (x["ts"] or datetime.min, x["id"]))

    profiles = {}
    pers = f["personas"]
    pers = pers[pers["network_id"] == network_id] if "network_id" in pers else pers
    for r in pers.itertuples(index=False):
        name = getattr(r, "display_name", None) or getattr(r, "name", None)
        if not name:
            continue
        d = {k: getattr(r, k) for k in pers.columns if hasattr(r, k)}
        d["Title"] = d.get("occupation")
        profiles[name] = d
    _attach_roles(messages, {k: {"Title": v.get("occupation")}
                             for k, v in profiles.items()})
    return normalize(messages), profiles


def load_socialmem_questions(network_id, _cache={}):
    """Return question dicts for one network."""
    import json as _json
    if "f" not in _cache:
        _cache["f"] = _social_frames()
    qa = _cache["f"]["qa"]
    qa = qa[qa["network_id"] == network_id]
    out = []
    for r in qa.itertuples(index=False):
        opts = None
        if r.answer_format == "multiple_choice" and r.options_json:
            try:
                raw = _json.loads(r.options_json)
                opts = raw if isinstance(raw, dict) else {
                    chr(65 + i): v for i, v in enumerate(raw)}
            except (ValueError, TypeError):
                opts = None
        # Gold evidence anchors -> turn ids, so the oracle upper bound works.
        refs = []
        try:
            for a in _json.loads(r.evidence_anchors_json or "[]"):
                if a.get("turn_id"):
                    refs.append(a["turn_id"])
        except (ValueError, TypeError):
            pass
        qtype = SOCIAL_QTYPES.get(str(r.query_type)[:2], str(r.query_type))
        out.append({
            "id": r.qa_id,
            "question": r.question,
            "answer": (r.correct_option if opts else r.answer),
            "options": opts,
            "reference": refs,          # list of turn ids (see _oracle_retrieve)
            "qtype": qtype,
            "category": qtype,
            "dimension": "Social Memory",
            "difficulty": r.difficulty,
        })
    return out


def list_socialmem_networks(_cache={}):
    if "f" not in _cache:
        _cache["f"] = _social_frames()
    return sorted(_cache["f"]["networks"]["network_id"].unique().tolist())
