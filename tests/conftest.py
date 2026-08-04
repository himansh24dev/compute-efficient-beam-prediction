import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.config import load_data_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_data_config("configs/data.yaml")
