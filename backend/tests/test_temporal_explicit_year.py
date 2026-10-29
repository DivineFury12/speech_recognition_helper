import datetime
import unittest

from src.services.temporal import parse_date


class ExplicitYearDatesTest(unittest.TestCase):
    def test_named_month_preserves_explicit_year(self):
        today = datetime.date(2026, 10, 29)
        cases = {
            '8 марта 2027 года': datetime.date(2027, 3, 8),
            '23 марта 2027 года': datetime.date(2027, 3, 23),
            '29 октября 2027 года': datetime.date(2027, 10, 29),
            '10 ноября 2027 года': datetime.date(2027, 11, 10),
            '26 ноября 2027 года': datetime.date(2027, 11, 26),
            '20 декабря 2027 года': datetime.date(2027, 12, 20),
            '31 декабря 2027 года': datetime.date(2027, 12, 31),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_date(phrase, today), expected)

    def test_implicit_year_still_uses_next_occurrence(self):
        today = datetime.date(2026, 10, 29)
        self.assertEqual(parse_date('8 марта', today), datetime.date(2027, 3, 8))
        self.assertEqual(parse_date('10 ноября', today), datetime.date(2026, 11, 10))
        self.assertEqual(parse_date('5 мая', today), datetime.date(2027, 5, 5))

    def test_named_month_rejects_invalid_day(self):
        self.assertIsNone(parse_date('31 ноября 2027 года', datetime.date(2026, 10, 29)))


if __name__ == '__main__':
    unittest.main()
