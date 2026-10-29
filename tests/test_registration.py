import json

import pytest

from gate0.config import check_lock, write_lock


def test_lock_covers_code_and_detects_drift(tmp_path):
    lock = tmp_path / "registration.lock"
    hashes = write_lock(lock)
    assert any(k.startswith("src/") for k in hashes) and any(k.startswith("prompts/") for k in hashes)
    assert check_lock(lock) == hashes["registration"]
    tampered = {**hashes, "src/moments.py": "0" * 64}
    lock.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="src/moments.py"):
        check_lock(lock)
