"""
Agent tools.

get_schedule(query, date=None, start_time=None, end_time=None) -> RAG retrieval over the
    vector store, optionally filtered to a specific date.

update_schedule(action, ...) -> add / update / remove an event. Writes go to the JSON
    system-of-record AND to the vector index, so retrieval always reflects the latest state.
"""
import json
import re
import uuid
from datetime import datetime, timedelta

from vector_store import SCHEDULE_JSON, event_to_text

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _load_events():
    with open(SCHEDULE_JSON) as f:
        return json.load(f)


def _save_events(events):
    events.sort(key=lambda e: (e["date"], e["start_time"]))
    with open(SCHEDULE_JSON, "w") as f:
        json.dump(events, f, indent=2)


def resolve_date_phrase(phrase, today=None):
    """Turn 'tomorrow', 'friday', 'next monday', 'August 15', '2026-08-20' etc.
    into a YYYY-MM-DD string. Returns None if it can't confidently resolve."""
    if not phrase:
        return None
    today = today or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    p = phrase.strip().lower()

    if p in ("today",):
        return today.strftime("%Y-%m-%d")
    if p in ("tomorrow",):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if p in ("yesterday",):
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.match(r"^\d{4}-\d{2}-\d{2}$", p)
    if m:
        return p

    # "next <weekday>" or just "<weekday>" -> nearest upcoming occurrence
    m = re.match(r"^(next\s+)?(" + "|".join(WEEKDAYS) + r")$", p)
    if m:
        target = WEEKDAYS.index(m.group(2))
        days_ahead = (target - today.weekday()) % 7
        if days_ahead == 0 and m.group(1):  # "next monday" said on a monday -> next week
            days_ahead = 7
        elif days_ahead == 0:
            days_ahead = 0  # "monday" said on monday -> today
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "August 15", "Aug 15", "15 August" (assume current/next occurrence's year)
    m = re.match(r"^([a-zA-Z]+)\s+(\d{1,2})$", p) or re.match(r"^(\d{1,2})\s+([a-zA-Z]+)$", p)
    if m:
        g1, g2 = m.group(1), m.group(2)
        month_str, day_str = (g1, g2) if not g1.isdigit() else (g2, g1)
        try:
            parsed = datetime.strptime(f"{month_str} {day_str} {today.year}", "%B %d %Y")
        except ValueError:
            try:
                parsed = datetime.strptime(f"{month_str} {day_str} {today.year}", "%b %d %Y")
            except ValueError:
                return None
        if parsed.date() < today.date():
            parsed = parsed.replace(year=today.year + 1)
        return parsed.strftime("%Y-%m-%d")

    return None


def parse_time_phrase(phrase):
    """'3 PM' / '15:00' / '3:30pm' -> 'HH:MM' (24h)."""
    if not phrase:
        return None
    p = phrase.strip().lower().replace(" ", "")
    m = re.match(r"^(\d{1,2}):?(\d{2})?(am|pm)?$", p)
    if not m:
        return None
    hour, minute, meridiem = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


# ---------------------------------------------------------------------------
# Tool 1: get_schedule  (RAG retrieval)
# ---------------------------------------------------------------------------
def get_schedule(store, query="", date=None, start_time=None, end_time=None, top_k=8):
    """
    Retrieve relevant schedule entries.
    - `query`: free-text semantic query (e.g. "client meetings", "what's on my plate")
    - `date`: exact YYYY-MM-DD to filter to a single day (most reliable path)
    - `start_time`/`end_time`: optional HH:MM window filter within that date
    Returns: list of event dicts, sorted by start_time.
    """
    where = {"date": date} if date else None
    results = store.query(query or "schedule", n_results=top_k, where=where)

    if start_time or end_time:
        def in_window(e):
            s = e["start_time"]
            e_end = e["end_time"]
            if start_time and e_end <= start_time:
                return False
            if end_time and s >= end_time:
                return False
            return True
        results = [e for e in results if in_window(e)]

    results.sort(key=lambda e: (e["date"], e["start_time"]))
    return results


def check_free_slot(store, date, start_time, end_time):
    """Convenience helper built on get_schedule: is [start_time,end_time) free on `date`?"""
    events = get_schedule(store, date=date, top_k=50)

    def overlaps(e):
        return not (e["end_time"] <= start_time or e["start_time"] >= end_time)
    conflicts = [e for e in events if overlaps(e)]
    return len(conflicts) == 0, conflicts


# ---------------------------------------------------------------------------
# Tool 2: update_schedule  (add / update / remove)
# ---------------------------------------------------------------------------
def update_schedule(store, action, event_id=None, date=None, start_time=None,
                     end_time=None, title=None, event_type="meeting", notes="",
                     duration_minutes=60):
    """
    action: "add" | "update" | "remove"

    add:    requires date, start_time, title. end_time derived from duration_minutes if absent.
    update: requires event_id OR (date + start_time) to locate the event; any of
            date/start_time/end_time/title/notes may be changed.
    remove: requires event_id OR (date + start_time) to locate the event.

    Returns: dict describing what happened (for the agent to report back to the user).
    """
    events = _load_events()

    if action == "add":
        if not (date and start_time and title):
            return {"status": "error", "message": "add requires date, start_time, and title"}
        if not end_time:
            end_dt = datetime.strptime(start_time, "%H:%M") + timedelta(minutes=duration_minutes)
            end_time = end_dt.strftime("%H:%M")
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        new_event = {
            "id": str(uuid.uuid4())[:8],
            "date": date,
            "day_of_week": day_name,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "type": event_type,
            "notes": notes,
        }
        events.append(new_event)
        _save_events(events)
        store.upsert(new_event)
        return {"status": "ok", "action": "added", "event": new_event}

    def locate(ev_list):
        if event_id:
            return next((e for e in ev_list if e["id"] == event_id), None)
        if date and start_time:
            return next((e for e in ev_list if e["date"] == date and e["start_time"] == start_time), None)
        return None

    if action == "update":
        target = locate(events)
        if not target:
            return {"status": "error", "message": "Could not find a matching event to update"}
        if title:
            target["title"] = title
        if notes:
            target["notes"] = notes
        if event_type:
            target["type"] = event_type
        # Move / reschedule
        new_date = date if date else target["date"]
        if start_time:
            duration = (
                datetime.strptime(target["end_time"], "%H:%M")
                - datetime.strptime(target["start_time"], "%H:%M")
            )
            target["start_time"] = start_time
            target["end_time"] = (datetime.strptime(start_time, "%H:%M") + duration).strftime("%H:%M")
        elif end_time:
            target["end_time"] = end_time
        target["date"] = new_date
        target["day_of_week"] = datetime.strptime(new_date, "%Y-%m-%d").strftime("%A")
        _save_events(events)
        store.upsert(target)
        return {"status": "ok", "action": "updated", "event": target}

    if action == "remove":
        target = locate(events)
        if not target:
            return {"status": "error", "message": "Could not find a matching event to remove"}
        events = [e for e in events if e["id"] != target["id"]]
        _save_events(events)
        store.delete(target["id"])
        return {"status": "ok", "action": "removed", "event": target}

    return {"status": "error", "message": f"Unknown action '{action}'"}
