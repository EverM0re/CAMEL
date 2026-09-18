"""Deterministic date arithmetic for duration questions.

EverMemBench's Temporal sub-task stays broken even in the paper's oracle
setting (best model 60%, LLaMA-4 34%) because the failure is arithmetic, not
retrieval: the model must count calendar days *or* working days between two
anchors, and weekends carry no dialogue at all. Asking an LLM to do the
subtraction in prose is unreliable, so we compute it and hand the model the
result as an explicit hint.
"""
import re
from datetime import date, datetime, timedelta

_DATE_PAT = re.compile(
    r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})"
    r"|\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(20\d{2})\b",
    re.IGNORECASE,
)
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

# Questions that ask for a DATE, not a span. GroupMemBench's temporal items are
# mostly of this shape ("what is the new deadline"), and its gold answers are
# absolute dates that the text states relatively ("by EOD Friday").
DATE_QUESTION_RE = re.compile(
    r"\bwhen\b|\bwhat date\b|\bwhich date\b|\bdeadline\b|\bdue\b|"
    r"\bscheduled\b|\bfreeze\b|\bcut ?off\b|\bby when\b|"
    r"\bwhat day\b|\bmoved to\b|\brescheduled\b",
    re.IGNORECASE)

WORKING_DAY_RE = re.compile(r"\bworking\s+days?\b|\bbusiness\s+days?\b|工作日",
                            re.IGNORECASE)
DURATION_RE = re.compile(
    r"\bhow\s+(?:many|long)\b|\bduration\b|\bhow\s+much\s+time\b|"
    r"\bdays?\s+(?:did|passed|elapsed|between)\b|多少天|多长时间",
    re.IGNORECASE)


def extract_dates(text):
    """Return the distinct dates mentioned in `text`, in order of appearance."""
    out = []
    for m in _DATE_PAT.finditer(text or ""):
        try:
            if m.group(1):
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                d = date(int(m.group(6)), _MONTHS[m.group(4).lower()],
                         int(m.group(5)))
        except (ValueError, KeyError):
            continue
        if d not in out:
            out.append(d)
    return out


def calendar_days(d0, d1):
    """Inclusive-of-both-endpoints span, matching the benchmark's convention.

    EverMemBench's worked example calls 2025-04-04 -> 2025-04-10 "7 days",
    i.e. (d1 - d0) + 1.
    """
    a, b = sorted((d0, d1))
    return (b - a).days + 1


def working_days(d0, d1):
    """Count weekdays in the inclusive span [d0, d1]."""
    a, b = sorted((d0, d1))
    n = 0
    cur = a
    while cur <= b:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def is_duration_question(question):
    return bool(DURATION_RE.search(question or ""))


def is_date_question(question):
    """True when the answer is a specific date rather than a span."""
    q = question or ""
    return bool(DATE_QUESTION_RE.search(q)) and not is_duration_question(q)


# Relative expressions that a message may use to name a date other than its own
# posting date. Resolving these is what the answer model gets wrong on its own:
# it reads the posting timestamp and reports that instead.
_REL_RE = re.compile(
    r"\b(?:by\s+|due\s+|on\s+)?(?:EOD\s+|end of day\s+|COB\s+)?"
    r"(today|tomorrow|yesterday|this (?:coming )?\w+day|next \w+day|"
    r"(?:this |next )?(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day|"
    r"in \d+ (?:days?|weeks?))\b",
    re.IGNORECASE)


def resolved_dates(messages, limit=12):
    """Resolve relative date expressions in each message against its own stamp.

    Returns [(iso_date, source_expression, message)] for the messages that name
    a date relatively. A message posted on 2025-07-17 saying "freeze by EOD
    Friday" resolves to 2025-07-18; the answer model, shown only the posting
    stamp, reports 2025-07-17 -- which is the single most common error we make
    on date questions.
    """
    out = []
    for m in messages:
        ts = m.get("ts")
        if not ts:
            continue
        for mt in _REL_RE.finditer(m.get("text") or ""):
            iso = resolve_relative(mt.group(1), ts)
            if iso:
                out.append((iso, mt.group(0).strip(), m))
                break
        if len(out) >= limit:
            break
    return out


def date_resolution_hint(question, evidence_msgs, limit=8):
    """Hint block for questions whose answer is a date stated relatively."""
    if not is_date_question(question):
        return ""
    res = resolved_dates(evidence_msgs, limit=limit)
    if not res:
        return ""
    lines = ["RESOLVED DATES (computed from each message's own timestamp; "
             "prefer these over the posting timestamp):"]
    for iso, expr, m in res[:limit]:
        stamp = m["ts"].strftime("%Y-%m-%d")
        author = m.get("author", "?")
        lines.append(f"- {author} wrote on {stamp}: \"{expr}\" -> {iso}")
    lines.append("- A relative expression refers to the resolved date above, "
                 "NOT to the date the message was posted.")
    return "\n".join(lines)


def wants_working_days(question):
    return bool(WORKING_DAY_RE.search(question or ""))


def date_hint(question, evidence_msgs, max_dates=8):
    """Build an arithmetic hint block for a duration question.

    Returns "" when the question is not a duration question or no candidate
    dates were retrieved. Otherwise returns a short block listing the candidate
    anchor dates and the precomputed span between the earliest and latest, in
    the unit the question asked for.
    """
    if not is_duration_question(question):
        return ""
    dates = []
    for m in evidence_msgs:
        ts = m.get("ts")
        if ts and ts.date() not in dates:
            dates.append(ts.date())
    dates.extend(d for d in extract_dates(question) if d not in dates)
    if len(dates) < 2:
        return ""
    dates.sort()
    lo, hi = dates[0], dates[-1]
    wd = wants_working_days(question)
    lines = ["DATE ARITHMETIC (computed, use these numbers -- do not recompute):"]
    lines.append(f"- Question asks for {'WORKING' if wd else 'calendar'} days.")
    shown = dates[:max_dates]
    lines.append("- Candidate anchor dates in the retrieved evidence: "
                 + ", ".join(f"{d.isoformat()} ({d.strftime('%a')})" for d in shown))
    lines.append(f"- Earliest {lo.isoformat()} -> latest {hi.isoformat()}: "
                 f"{calendar_days(lo, hi)} calendar days, "
                 f"{working_days(lo, hi)} working days.")
    if len(shown) >= 2:
        lines.append("- Pairwise spans (calendar / working):")
        for i in range(len(shown)):
            for j in range(i + 1, len(shown)):
                a, b = shown[i], shown[j]
                lines.append(f"    {a.isoformat()} -> {b.isoformat()}: "
                             f"{calendar_days(a, b)} / {working_days(a, b)}")
    lines.append("- Pick the pair matching the true start and genuine completion "
                 "anchors, then report the span in the requested unit.")
    return "\n".join(lines)


def resolve_relative(expr, anchor_ts):
    """Resolve 'EOD tomorrow' / 'next Friday' against an anchor timestamp.

    Returns an ISO date string, or None when the expression is not recognised.
    GroupMemBench temporal questions ground relative expressions on the anchor
    message's timestamp and expect an absolute YYYY-MM-DD answer.
    """
    if not anchor_ts:
        return None
    base = anchor_ts.date() if isinstance(anchor_ts, datetime) else anchor_ts
    e = (expr or "").lower()
    if "today" in e or "eod" in e and "tomorrow" not in e:
        return base.isoformat()
    if "tomorrow" in e:
        return (base + timedelta(days=1)).isoformat()
    if "yesterday" in e:
        return (base - timedelta(days=1)).isoformat()
    days = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]
    for i, d in enumerate(days):
        if d in e:
            delta = (i - base.weekday()) % 7
            if "next" in e:
                delta = delta + 7 if delta == 0 else delta
            elif delta == 0:
                delta = 0
            return (base + timedelta(days=delta)).isoformat()
    m = re.search(r"in\s+(\d+)\s+days?", e)
    if m:
        return (base + timedelta(days=int(m.group(1)))).isoformat()
    m = re.search(r"in\s+(\d+)\s+weeks?", e)
    if m:
        return (base + timedelta(weeks=int(m.group(1)))).isoformat()
    return None
