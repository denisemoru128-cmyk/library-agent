from datetime import date

from library_agent import dates


def test_the_weekday_is_worked_out_from_the_date():
    assert dates.short(date(2026, 11, 21)) == "Sat 21 Nov"
    assert dates.full(date(2026, 11, 21)) == "Saturday 21 November 2026"


def test_days_have_no_leading_zero():
    assert dates.short(date(2026, 12, 5)) == "Sat 5 Dec"