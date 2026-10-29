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


def test_registration_time_is_recorded_but_not_hashed(tmp_path):
    from gate0.config import registered_at
    lock = tmp_path / "registration.lock"
    write_lock(lock)
    assert registered_at(lock) > 0
    check_lock(lock)  # the timestamp is not part of the registered object


def test_persona_text_never_names_a_construct():
    """v1 lesson: naming the construct invites recalled literature instead of an induced belief."""
    from gate0.config import PROMPTS_DIR
    banned = ("conservat", "representativ", "overconfiden", "self-attribution", "self attribution", "bias",
              "momentum", "heuristic")  # "anchored" (v1 conservatism level 3) is kept: plain English
    for f in [*PROMPTS_DIR.glob("*.txt"), *PROMPTS_DIR.glob("v2b/*.txt")]:
        if f.stem in ("market_task", "market_task_masked"):
            continue
        text = f.read_text().lower()
        assert not [w for w in banned if w in text], f
