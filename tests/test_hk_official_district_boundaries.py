from ghost_districts import get_district


def test_hk_island_uses_official_district_polygons():
    assert get_district(22.287, 114.213) == ("Eastern", "Hong Kong Island")
    assert get_district(22.2855, 114.1580) == (
        "Central & Western",
        "Hong Kong Island",
    )


def test_kowloon_uses_official_district_polygons():
    assert get_district(22.3182, 114.1685) == ("Yau Tsim Mong", "Kowloon West")
    assert get_district(22.300, 114.220) == ("Kwun Tong", "Kowloon East")


def test_new_territories_and_islands_use_official_district_polygons():
    assert get_district(22.381, 114.188) == ("Sha Tin", "New Territories South")
    assert get_district(22.308, 113.9185) == ("Islands", "New Territories South")
