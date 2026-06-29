import ghost_listener


def test_iter_grid_points_respects_max_cells():
    points = list(ghost_listener.iter_grid_points(max_cells=3))

    assert points == [
        (ghost_listener.HK_LAT_MIN, ghost_listener.HK_LNG_MIN),
        (ghost_listener.HK_LAT_MIN, 113.88),
        (ghost_listener.HK_LAT_MIN, 113.93),
    ]


def test_run_once_can_skip_active_sweep(monkeypatch):
    calls = []

    monkeypatch.setattr(ghost_listener, "poll_app_checking", lambda store: calls.append("app"))
    monkeypatch.setattr(ghost_listener, "poll_news", lambda store: calls.append("news"))
    monkeypatch.setattr(ghost_listener, "poll_notification_record", lambda store: calls.append("notifications"))
    monkeypatch.setattr(
        ghost_listener,
        "poll_nearby_alerts",
        lambda store, max_grid_cells=None: calls.append(("24h", max_grid_cells)),
    )
    monkeypatch.setattr(
        ghost_listener,
        "poll_nearby_active",
        lambda store, max_grid_cells=None: calls.append(("active", max_grid_cells)),
    )

    store = {"alerts": {}, "meta": {}}
    ghost_listener.run_once(store, include_active=False, max_grid_cells=48)

    assert calls == ["app", "news", "notifications", ("24h", 48)]
    assert store["meta"]["total_alerts"] == 0
