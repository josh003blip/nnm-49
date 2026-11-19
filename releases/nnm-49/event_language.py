"""A deterministic English-to-CIL bridge for same-day event questions."""

from dataclasses import dataclass, field
import re

from cil import compile_cil, execute


ACTOR = r"[a-z][a-z'’-]{1,30}"
ACTION = r"[a-z][a-z'’-]{1,24}ing"
TIME = r"(?:[0-9]{1,2}:[0-9]{2}(?: ?(?:am|pm))?|[0-9]{1,2} ?(?:am|pm))"
SUBJECT = rf"(?P<actor>{ACTOR})"
ACTIVITY = rf"(?P<action>{ACTION})"
START = rf"(?P<start>{TIME})"
END = rf"(?P<end>{TIME})"
QUERY = rf"(?P<query>{TIME})"
AUX = r"(?:was|were|is|are)"
PRONOUNS = {"he", "she", "they"}
IGNORE_ACTORS = PRONOUNS | {"it", "we", "i", "you"}

INTERVAL_PATTERNS = [
    rf"{SUBJECT} {AUX} {ACTIVITY} from {START} (?:to|until) {END}",
    rf"{SUBJECT} {AUX} {ACTIVITY} between {START} and {END}",
    rf"from {START} (?:to|until) {END},? {SUBJECT} {AUX} {ACTIVITY}",
    rf"between {START} and {END},? {SUBJECT} {AUX} {ACTIVITY}",
    rf"{SUBJECT} (?:started|began) {ACTIVITY} at {START} and (?:stopped|finished|ended) at {END}",
]
START_PATTERNS = [
    rf"{SUBJECT} (?:started|began) {ACTIVITY} at {START}",
    rf"{SUBJECT} {AUX} {ACTIVITY} from {START}",
]
END_PATTERNS = [
    rf"{SUBJECT} (?:stopped|finished|ended) {ACTIVITY} at {END}",
    rf"{SUBJECT} {AUX} {ACTIVITY} until {END}",
]
QUERY_PATTERNS = [
    rf"at {QUERY},? {AUX} {SUBJECT} (?:still )?{ACTIVITY}",
    rf"{AUX} {SUBJECT} (?:still )?{ACTIVITY} at {QUERY}",
]
UNTIMED_QUERIES = [rf"{AUX} {SUBJECT} (?:still )?{ACTIVITY}"]
FOLLOW_END = rf"{SUBJECT} (?:stopped|finished|ended) at {END}"
FOLLOW_START = rf"{SUBJECT} (?:started|began) at {START}"
FOLLOW_QUERY = rf"at {QUERY}"


@dataclass
class Event:
    actor: str
    action: str
    start: set[str] = field(default_factory=set)
    end: set[str] = field(default_factory=set)


def normalize(text):
    corrected = re.sub(
        r"\b([ap])\s*\.\s*m(?:\.(?=\s*(?:[,?!;]|(?:to|and|until|was|were|is|are)\b)))?",
        r"\1m", text.casefold(),
    )
    return re.sub(r"\s+", " ", corrected).strip()


def match(patterns, text):
    for pattern in patterns:
        if found := re.fullmatch(pattern, text):
            return found.groupdict()
    return None


def clock(text):
    parts = re.fullmatch(r"(\d{1,2})(?::(\d{2}))? ?(am|pm)?", text)
    if not parts:
        raise ValueError("Invalid clock time")
    raw_hour, raw_minute, marker = parts.groups()
    hour, minute = int(raw_hour), int(raw_minute or 0)
    if minute >= 60 or (marker and not 1 <= hour <= 12) or (not marker and hour >= 24):
        raise ValueError("Invalid clock time")
    if marker:
        return ((hour % 12) + (12 if marker == "pm" else 0)) * 60 + minute, True
    return hour * 60 + minute, not 1 <= hour <= 12


def civil_time(minutes):
    return f"{minutes // 60}:{minutes % 60:02d}"


def clarification(slot, prompt, **details):
    return dict(status="ok", disposition="clarify", missing=slot, answer=prompt, **details)


def abstain(reason):
    return dict(status="ok", disposition="unsupported", reason=reason,
                answer="I couldn't establish the event and its timing from that description.")


def evaluate_interval(record):
    start, end, query = (record[key] for key in ("start_minute", "end_minute", "query_minute"))
    # The CIL VM calculates both margins. Interval truth uses [start, end).
    elapsed = execute(compile_cil(f"(program (add x (mul -1 {start})))"), query, None)
    remaining = execute(compile_cil(f"(program (add {end} (mul -1 x)))"), query, None)
    return elapsed >= 0 and remaining > 0, {"elapsed_minutes": elapsed, "remaining_minutes": remaining}


def answer_event(text, follow_up=None):
    question = normalize(text)
    chunks = [item.strip() for item in re.split(r"[.!?]+", question) if item.strip()]
    if not question.endswith("?") or question.count("?") != 1 or len(chunks) < 1:
        return abstain("expected_one_event_question")
    prompt = match(QUERY_PATTERNS, chunks[-1]) or match(UNTIMED_QUERIES, chunks[-1])
    if prompt is None:
        return abstain("unsupported_question")
    facts = chunks[:-1]
    if follow_up:
        follow_up = normalize(follow_up)
        if lone_time := re.fullmatch(rf"(?:at )?(?P<time>{TIME})[.!]?", follow_up):
            pending = answer_event(text)
            value = lone_time["time"]
            if pending.get("missing") in ("start", "end"):
                edge = "started" if pending["missing"] == "start" else "stopped"
                follow_up = f"{pending['actor']} {edge} {pending['action']} at {value}"
            elif pending.get("missing") == "query":
                follow_up = f"at {value}"
        facts.extend(item.strip() for item in re.split(r"[.!?]+", follow_up) if item.strip())

    events = {}
    for statement in facts:
        if (untimed := match([FOLLOW_QUERY], statement)) and not prompt.get("query"):
            prompt["query"] = untimed["query"]
            continue
        evidence = match(INTERVAL_PATTERNS + START_PATTERNS + END_PATTERNS, statement)
        if evidence is not None:
            actor, action = evidence["actor"], evidence["action"]
            if actor in PRONOUNS:
                prior = [event for event in events.values() if event.action == action]
                if len(prior) != 1:
                    return clarification("actor", f"Who was {action}?")
                actor = prior[0].actor
            elif actor in IGNORE_ACTORS:
                return clarification("actor", f"Who was {action}?")
            event = events.setdefault((actor, action), Event(actor, action))
        else:
            evidence = match([FOLLOW_START, FOLLOW_END], statement)
            if evidence is None:
                return abstain("unsupported_statement")
            choices = [event for event in events.values() if event.action == prompt["action"] and
                       (event.actor == evidence["actor"] or evidence["actor"] in PRONOUNS)]
            if len(choices) != 1:
                return clarification("actor", f"Whose {prompt['action']} interval are you asking about?")
            event = choices[0]
        for field_name in ("start", "end"):
            if evidence.get(field_name):
                getattr(event, field_name).add(evidence[field_name])

    actor, action = prompt["actor"], prompt["action"]
    if actor in PRONOUNS:
        candidates = [event for event in events.values() if event.action == action]
        if len(candidates) != 1:
            return clarification("actor", f"Who was {action}?")
        event = candidates[0]
    elif actor in IGNORE_ACTORS:
        return clarification("actor", f"Who was {action}?")
    else:
        event = events.get((actor, action), Event(actor, action))

    who = event.actor.capitalize()
    for edge, verb in (("start", "start"), ("end", "stop")):
        readings = getattr(event, edge)
        if not readings:
            return clarification(edge, f"When did {who} {verb} {action}?", actor=event.actor, action=action)
    if not prompt.get("query"):
        return clarification("query", "What time should I check?")
    try:
        start_times = {clock(value) for value in event.start}
        end_times = {clock(value) for value in event.end}
        query, query_explicit = clock(prompt["query"])
    except ValueError:
        return clarification("clock", "Could you give valid clock times for the event and question?")
    if len(start_times) != 1 or len(end_times) != 1:
        return clarification("conflict", f"What was the correct time interval for {who}'s {action}?")
    start, start_explicit = next(iter(start_times))
    end, end_explicit = next(iter(end_times))
    signatures = {start_explicit, end_explicit, query_explicit}
    if signatures == {False, True}:
        return clarification("clock", "Could you give all three times with AM or PM, or use 24-hour time?")
    if start >= end:
        return clarification("interval", f"Did {who}'s {action} cross midnight, or is one of the times wrong?")

    record = dict(type="cil.event_question.v1", actor=event.actor, action=action,
                  start_minute=start, end_minute=end, query_minute=query,
                  end_exclusive=True)
    happening, trace = evaluate_interval(record)
    event_times = f"{civil_time(start)} to {civil_time(end)}"
    checked = civil_time(query)
    if happening:
        response = f"Yes. {who} was {action} at {checked}, during the {event_times} interval."
    else:
        location = "before the event started" if query < start else "at or after the event ended"
        response = f"No. {who}'s {action} interval was {event_times}, and {checked} was {location}."
    return dict(status="ok", disposition="answered", happening=happening, answer=response,
                record=record, cil_trace=trace)
