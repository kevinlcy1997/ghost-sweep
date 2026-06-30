from ghost_districts import get_district


def test_get_district_does_not_assign_eastern_north_of_harbour():
    district, region = get_district(22.300, 114.220)

    assert district != "Eastern"
    assert region == "Kowloon East"


def test_get_district_keeps_eastern_on_hk_island():
    district, region = get_district(22.287, 114.213)

    assert district == "Eastern"
    assert region == "Hong Kong Island"
