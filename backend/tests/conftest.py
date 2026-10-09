import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ASR_ENABLED"] = "false"
os.environ["LLM_PROVIDER"] = "rules"
os.environ["DB_PATH"] = ":memory:"


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c
