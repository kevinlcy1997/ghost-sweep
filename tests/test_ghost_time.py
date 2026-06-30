from datetime import datetime
from importlib.util import find_spec

from analysis.build_fixed_grid_feature_mart import parse_timestamp


def test_source_create_dt_is_interpreted_as_hong_kong_clock_time():
    assert find_spec("ghost_time") is not None
    from ghost_time import parse_hk_source_time, to_hk_feature_time

    parsed = parse_hk_source_time("2026-06-28 17:24:28")

    assert parsed.isoformat() == "2026-06-28T17:24:28+08:00"
    assert to_hk_feature_time("2026-06-28 17:24:28") == datetime(2026, 6, 28, 17, 24, 28)


def test_utc_iso_timestamp_is_converted_to_hong_kong_feature_time():
    parsed = parse_timestamp("2026-06-29T06:09:24+00:00")

    assert parsed == datetime(2026, 6, 29, 14, 9, 24)


def test_forecast_target_time_uses_hong_kong_clock(monkeypatch):
    import ghost_predict

    assert hasattr(ghost_predict, "get_forecast_target_time")

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 29, 6, 9, 24, tzinfo=tz)

    monkeypatch.setattr("ghost_predict.datetime", FrozenDatetime)

    assert ghost_predict.get_forecast_target_time() == datetime(2026, 6, 29, 14, 9, 24)
