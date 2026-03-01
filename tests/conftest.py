from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_dashmanifest_xml() -> bytes:
    return (FIXTURES_DIR / "sample_dashmanifest.xml").read_bytes()
