"""
The agent: decides, per user message, whether to call get_schedule, update_schedule,
both, or neither — then composes a natural-language reply.

Two modes:
  1. LLM mode (preferred): if ANTHROPIC_API_KEY is set, uses Claude's native tool-use
     (the Messages API) to let the model itself choose the tool and arguments. This is
     the "real" agentic pipeline the assignment describes.
  2. Fallback mode: a small deterministic intent parser (regex/keyword based) covering
     the example query types, so the whole app works end-to-end with zero external
     services for local dev/demo/grading.

Both modes call the exact same tools.get_schedule / tools.update_schedule functions,
so retrieval and writes are always backed by the same RAG vector store.
"""
import json
import os
import re
from datetime import datetime, timedelta

import requests

import tools
from vector_store import get_vector_store, build_index_from_schedule, event_to_text

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

TOOL_DEFINITIONS = [
    {
        "name": "get_schedule",
        "description": (
            "Retrieve relevant schedule entries (meetings, workshops, tasks, appointments) "
            "using semantic search over the schedule, optionally filtered to a specific date "
            "and/or time window. Use this for any question about what is scheduled, whether "
            "the user is free/busy, or to look up an event before updating it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search, e.g. 'client meetings' or 'anything today'"},
                "date": {"type": "string", "description": "Exact date filter as YYYY-MM-DD, if the user referenced a specific day"},
                "start_time": {"type": "string", "description": "HH:MM (24h) start of a time window filter, if relevant"},
                "end_time": {"type": "string", "description": "HH:MM (24h) end of a time window filter, if relevant"},
            },
            "required": [],
        },
    },
    {
        "name": "update_schedule",
        "description": (
            "Add, update (including moving/rescheduling), or remove a schedule entry. "
            "For update/remove, identify the target event first (use get_schedule if you "
            "don't already know its date/start_time)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "remove"]},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM (24h), also used to locate the event for update/remove"},
                "end_time": {"type": "string", "description": "HH:MM (24h), optional"},
                "title": {"type": "string"},
                "event_type": {"type": "string", "enum": ["meeting", "workshop", "task", "appointment"]},
                "notes": {"type": "string"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["action"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful schedule assistant. Today's date is {today} ({today_weekday}). "
    "You manage the user's calendar for the next 30 days using two tools: get_schedule "
    "(retrieval) and update_schedule (add/update/remove). Always resolve relative dates "
    "('tomorrow', 'Friday') to YYYY-MM-DD yourself before calling a tool. Prefer calling "
    "get_schedule before update_schedule when you need to locate an existing event (e.g. "
    "'move my meeting from 2pm to 4pm' — first find the 2pm event today). After tool "
    "results come back, answer the user directly and concisely in plain language — don't "
    "mention tool names or internal mechanics."
)


class Agent:
    def __init__(self):
        self.store = get_vector_store()
        if not self.store.all_events():
            build_index_from_schedule(self.store)

    # -------------------------------------------------------------- dispatch
    def _run_tool(self, name, args):
        if name == "get_schedule":
            results = tools.get_schedule(
                self.store,
                query=args.get("query", ""),
                date=args.get("date"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
            )
            return {"count": len(results), "events": results}
        if name == "update_schedule":
            return tools.update_schedule(
                self.store,
                action=args.get("action"),
                date=args.get("date"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
                title=args.get("title"),
                event_type=args.get("event_type", "meeting"),
                notes=args.get("notes", ""),
                duration_minutes=args.get("duration_minutes", 60),
            )
        return {"error": f"unknown tool {name}"}

    def handle_message(self, user_message, history=None):
        if ANTHROPIC_API_KEY:
            try:
                return self._handle_with_llm(user_message, history or [])
            except Exception as exc:  # network issues, bad key, etc. -> graceful fallback
                fallback = self._handle_with_rules(user_message)
                return fallback + f"\n\n_(LLM tool-use unavailable, used rule-based fallback: {exc})_"
        return self._handle_with_rules(user_message)

    # -------------------------------------------------------------- LLM mode
    def _handle_with_llm(self, user_message, history):
        today = datetime.now()
        system = SYSTEM_PROMPT.format(today=today.strftime("%Y-%m-%d"), today_weekday=today.strftime("%A"))
        messages = list(history) + [{"role": "user", "content": user_message}]
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        for _ in range(5):  # tool-use loop guard
            payload = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "system": system,
                "tools": TOOL_DEFINITIONS,
                "messages": messages,
            }
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data["content"]
            messages.append({"role": "assistant", "content": content})

            tool_calls = [b for b in content if b["type"] == "tool_use"]
            if not tool_calls:
                text_blocks = [b["text"] for b in content if b["type"] == "text"]
                return "\n".join(text_blocks).strip()

            tool_results = []
            for call in tool_calls:
                result = self._run_tool(call["name"], call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})

        return "I ran into trouble completing that request — could you rephrase it?"

    # -------------------------------------------------------------- fallback (rule-based) mode
    def _handle_with_rules(self, msg):
        m = msg.strip().lower()

        # --- "free" / availability check ---
        if "free" in m or "available" in m:
            date = self._extract_date(m) or datetime.now().strftime("%Y-%m-%d")
            start, end = self._extract_time_window(m)
            free, conflicts = tools.check_free_slot(self.store, date, start or "00:00", end or "23:59")
            if free:
                return f"Yes, you're free on {date}{self._window_desc(start, end)}."
            lines = [self._describe_event(e) for e in conflicts]
            return f"No — you have something scheduled on {date}{self._window_desc(start, end)}:\n" + "\n".join(lines)

        # --- add ---
        if any(k in m for k in ["add", "schedule a", "book", "create"]) and "meeting" in m or "add a" in m:
            return self._handle_add(msg)
        if re.search(r"\badd\b", m):
            return self._handle_add(msg)

        # --- move / update ---
        if any(k in m for k in ["move", "reschedule", "change", "shift", "push"]):
            return self._handle_move(msg)

        # --- remove ---
        if any(k in m for k in ["cancel", "remove", "delete"]):
            return self._handle_remove(msg)

        # --- default: retrieval ---
        date = self._extract_date(m)
        events = tools.get_schedule(self.store, query=msg, date=date)
        if not events:
            scope = f" on {date}" if date else ""
            return f"I don't see anything scheduled{scope}."
        header = f"Here's what you have{' on ' + date if date else ''}:"
        lines = [self._describe_event(e) for e in events]
        return header + "\n" + "\n".join(lines)

    # ---- helpers for rule-based mode ----
    def _extract_date(self, m):
        for phrase in ["today", "tomorrow", "yesterday"] + tools.WEEKDAYS + ["next " + d for d in tools.WEEKDAYS]:
            if phrase in m:
                d = tools.resolve_date_phrase(phrase)
                if d:
                    return d
        month_match = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})",
            m,
        )
        if month_match:
            return tools.resolve_date_phrase(f"{month_match.group(1)} {month_match.group(2)}")
        return None

    def _extract_time_window(self, m):
        if "morning" in m:
            return "06:00", "12:00"
        if "afternoon" in m:
            return "12:00", "17:00"
        if "evening" in m or "night" in m:
            return "17:00", "22:00"
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", m)
        if len(times) >= 2:
            return tools.parse_time_phrase(times[0]), tools.parse_time_phrase(times[1])
        if len(times) == 1:
            t = tools.parse_time_phrase(times[0])
            return t, (datetime.strptime(t, "%H:%M") + timedelta(hours=1)).strftime("%H:%M")
        return None, None

    @staticmethod
    def _window_desc(start, end):
        if start and end and not (start == "00:00" and end == "23:59"):
            return f" between {start} and {end}"
        return ""

    @staticmethod
    def _describe_event(e):
        return f"- [{e['type']}] {e['title']} — {e['date']} ({e['day_of_week']}) {e['start_time']}–{e['end_time']}"

    def _handle_add(self, msg):
        date = self._extract_date(msg.lower()) or datetime.now().strftime("%Y-%m-%d")
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", msg.lower())
        start_time = tools.parse_time_phrase(times[0]) if times else "09:00"
        title_match = re.search(r"(?:meeting|event|appointment|workshop|task)\s+(?:with|for|on|about)\s+(?:the\s+)?([^,.]+?)(?:\s+on\b|\s+at\b|$)", msg, re.I)
        title = title_match.group(1).strip() if title_match else "New Meeting"
        if title_match:
            title = ("Meeting with " + title) if "with" in msg.lower() else title.capitalize()
        result = tools.update_schedule(self.store, action="add", date=date, start_time=start_time,
                                        title=title, event_type="meeting")
        if result["status"] != "ok":
            return f"Couldn't add that: {result['message']}"
        e = result["event"]
        return f"Added: {self._describe_event(e)}"

    def _handle_move(self, msg):
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", msg.lower())
        date = self._extract_date(msg.lower()) or datetime.now().strftime("%Y-%m-%d")
        if len(times) < 2:
            return "Tell me both the current time and the new time, e.g. 'move my 2pm meeting to 4pm'."
        old_time = tools.parse_time_phrase(times[0])
        new_time = tools.parse_time_phrase(times[1])
        candidates = tools.get_schedule(self.store, date=date, start_time=old_time,
                                         end_time=(datetime.strptime(old_time, "%H:%M") + timedelta(minutes=1)).strftime("%H:%M"))
        if not candidates:
            return f"I couldn't find an event at {old_time} on {date} to move."
        target = candidates[0]
        result = tools.update_schedule(self.store, action="update", event_id=target["id"], start_time=new_time)
        if result["status"] != "ok":
            return f"Couldn't move that: {result['message']}"
        e = result["event"]
        return f"Moved '{e['title']}' to {e['start_time']}–{e['end_time']} on {e['date']}."

    def _handle_remove(self, msg):
        m = msg.lower()
        date = self._extract_date(m) or datetime.now().strftime("%Y-%m-%d")
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", m)
        if times:
            start_time = tools.parse_time_phrase(times[0])
            candidates = tools.get_schedule(self.store, date=date, start_time=start_time,
                                             end_time=(datetime.strptime(start_time, "%H:%M") + timedelta(minutes=1)).strftime("%H:%M"))
        else:
            candidates = tools.get_schedule(self.store, query=msg, date=date)
        if not candidates:
            return "I couldn't find a matching event to remove."
        target = candidates[0]
        result = tools.update_schedule(self.store, action="remove", event_id=target["id"])
        if result["status"] != "ok":
            return f"Couldn't remove that: {result['message']}"
        return f"Removed: {self._describe_event(result['event'])}"
