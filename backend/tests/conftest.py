import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ASR_ENABLED"] = "false"
os.environ["LLM_PROVIDER"] = "rules"
os.environ["DB_PATH"] = ":memory:"
os.environ["ACCESS_CODE"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["FRONTEND_DIST"] = ""


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app, headers={"X-Requested-With": "assistant"}) as c:
        yield c
