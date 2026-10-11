import os
import pathlib

_CONFIG = pathlib.Path(__file__).parent / "fixtures" / "config.yaml"
os.environ.setdefault("LLM_MITM_CONFIG", str(_CONFIG))
