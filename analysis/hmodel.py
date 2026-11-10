"""Runs analysis/harness.py with a thinking-off form registered for every model (Opus and Fable run adaptive only)."""
import runpy, sys
import gate0.config as C
_load = C.load_config
def load():
    cfg = _load()
    cfg.models.thinking_off.setdefault("claude-haiku-5-5", "disabled")
    return cfg
C.load_config = load
sys.argv = [sys.argv[0].replace("hmodel", "harness")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
