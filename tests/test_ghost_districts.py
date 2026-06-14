# tests/test_ghost_districts.py
import pytest
from ghost_districts import get_district


def test_mong_kok_district():
    district, region = get_district(22.3182, 114.1685)
    assert district == "Mong Kok"
    assert region == "Kowloon West"


def test_central_district():
    district, region = get_district(22.2855, 114.1580)
    assert district == "Central"
    assert region == "Hong Kong Island"


def test_ocean_fallback():
    district, region = get_district(22.30, 114.00)
    assert district != ""
    assert region != ""
