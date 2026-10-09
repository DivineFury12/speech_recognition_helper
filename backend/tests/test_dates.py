from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from assistant import dates

TODAY = date(2026, 10, 9)


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("сегодня", date(2026, 10, 9)),
        ("завтра", date(2026, 10, 10)),
        ("послезавтра", date(2026, 10, 11)),
        ("через 3 дня", date(2026, 10, 12)),
        ("через два дня", date(2026, 10, 11)),
        ("через неделю", date(2026, 10, 16)),
        ("в понедельник", date(2026, 10, 12)),
        ("в пятницу", date(2026, 10, 9)),
        ("в следующую пятницу", date(2026, 10, 16)),
        ("во вторник", date(2026, 10, 13)),
        ("12 октября", date(2026, 10, 12)),
        ("1 января", date(2027, 1, 1)),
        ("3 мая", date(2027, 5, 3)),
        ("15.11", date(2026, 11, 15)),
        ("2026-12-31", date(2026, 12, 31)),
    ],
)
def test_parse_date(phrase, expected):
    assert dates.parse_date(phrase, TODAY) == expected


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("в 15:00", time(15, 0)),
        ("в 9:30", time(9, 30)),
        ("в 10 утра", time(10, 0)),
        ("в 3 дня", time(15, 0)),
        ("в 7 вечера", time(19, 0)),
        ("в три часа дня", time(15, 0)),
        ("в 3", time(15, 0)),
        ("в 10", time(10, 0)),
        ("в 10.30", time(10, 30)),
        ("в полдень", time(12, 0)),
        ("вечером", time(19, 0)),
        ("15", time(15, 0)),
    ],
)
def test_parse_time(phrase, expected):
    assert dates.parse_time(phrase) == expected


def test_time_does_not_eat_dates():
    assert dates.find_time("встреча 12 октября") is None
    assert dates.find_time("через 2 дня") is None
    assert dates.find_date("в 10.30", TODAY) is None


def test_relative_datetime():
    now = datetime(2026, 10, 9, 23, 30, tzinfo=ZoneInfo("Europe/Moscow"))
    m = dates.find_relative_datetime("через 2 часа", now)
    assert m.value == datetime(2026, 10, 10, 1, 30, tzinfo=ZoneInfo("Europe/Moscow"))


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("на час", 60),
        ("на полчаса", 30),
        ("на полтора часа", 90),
        ("на 2 часа", 120),
        ("на 45 минут", 45),
    ],
)
def test_duration(phrase, expected):
    assert dates.find_duration(phrase).value == expected


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("что у меня сегодня", (date(2026, 10, 9), date(2026, 10, 9))),
        ("что у меня завтра", (date(2026, 10, 10), date(2026, 10, 10))),
        ("что у меня на неделе", (date(2026, 10, 9), date(2026, 10, 15))),
        ("на этой неделе", (date(2026, 10, 9), date(2026, 10, 11))),
        ("на следующей неделе", (date(2026, 10, 12), date(2026, 10, 18))),
        ("на выходных", (date(2026, 10, 10), date(2026, 10, 11))),
        (None, (date(2026, 10, 9), date(2026, 10, 9))),
    ],
)
def test_period(phrase, expected):
    assert dates.parse_period(phrase, TODAY) == expected
