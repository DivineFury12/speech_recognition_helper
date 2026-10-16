from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

RRULE = {
    "daily": "FREQ=DAILY",
    "weekdays": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    "weekly": "FREQ=WEEKLY",
    "monthly": "FREQ=MONTHLY",
}
KIND_PREFIX = {"task": "Задача", "reminder": "Напоминание", "meeting": "Встреча"}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> list[str]:
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return [line]
    out, chunk = [], ""
    for ch in line:
        if len((chunk + ch).encode("utf-8")) > (75 if not out else 74):
            out.append(chunk)
            chunk = ""
        chunk += ch
    out.append(chunk)
    return [out[0]] + [" " + c for c in out[1:]]


def _utc(entry: dict[str, Any]) -> datetime:
    try:
        tz = ZoneInfo(entry["timezone"])
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    local = datetime.combine(date.fromisoformat(entry["date"]), time.fromisoformat(entry["time"]), tz)
    return local.astimezone(timezone.utc)


def build_calendar(entries: list[dict[str, Any]], name: str = "Голосовой ассистент") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//speech-recognition-helper//assistant//RU",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_escape(name)}",
    ]
    for e in entries:
        if not e.get("date"):
            continue
        lines += ["BEGIN:VEVENT", f"UID:{e['id']}@speech-assistant", f"DTSTAMP:{stamp}"]
        if e.get("time"):
            start = _utc(e)
            minutes = e.get("duration_minutes") or (60 if e["kind"] == "meeting" else 15)
            lines += [
                f"DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}",
                f"DTEND:{(start + timedelta(minutes=minutes)).strftime('%Y%m%dT%H%M%SZ')}",
            ]
        else:
            d = date.fromisoformat(e["date"])
            lines += [f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}",
                      f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}"]
        done = "✓ " if e.get("status") == "completed" else ""
        lines.append(f"SUMMARY:{_escape(done + KIND_PREFIX[e['kind']] + ': ' + e['title'])}")
        if e.get("recurrence") in RRULE:
            rule = RRULE[e["recurrence"]]
            if e.get("recurrence_until"):
                rule += f";UNTIL={date.fromisoformat(e['recurrence_until']).strftime('%Y%m%d')}T235959Z"
            lines.append(f"RRULE:{rule}")
            for skipped in e.get("exdates") or []:
                day = date.fromisoformat(skipped)
                if e.get("time"):
                    stamp_skip = _utc({**e, "date": day.isoformat()}).strftime("%Y%m%dT%H%M%SZ")
                    lines.append(f"EXDATE:{stamp_skip}")
                else:
                    lines.append(f"EXDATE;VALUE=DATE:{day.strftime('%Y%m%d')}")
        if e.get("status") == "cancelled":
            lines.append("STATUS:CANCELLED")
        lead = e.get("remind_minutes")
        if lead is None and e["kind"] == "reminder":
            lead = 0
        if lead is not None and e.get("time"):
            trigger = "PT0M" if lead == 0 else f"-PT{lead}M"
            lines += ["BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_escape(e['title'])}",
                      f"TRIGGER:{trigger}", "END:VALARM"]
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    folded = [part for line in lines for part in _fold(line)]
    return "\r\n".join(folded) + "\r\n"
