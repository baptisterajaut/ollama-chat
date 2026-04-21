"""Pytest configuration: make the project importable when running from any cwd.

The project's dependencies (ollama, textual, openai, httpx) live in .venv. When
pytest is invoked from the pipx-managed global install, those imports would
fail — so we splice the venv's site-packages onto sys.path.
"""

import sys
import sysconfig
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add venv site-packages if present so pytest can import ollama/textual/openai/httpx
# even when invoked via a pipx-installed pytest binary.
_VENV = _PROJECT_ROOT / ".venv"
if _VENV.exists():
    py_major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _VENV_SITE = _VENV / "lib" / py_major_minor / "site-packages"
    if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
        sys.path.insert(0, str(_VENV_SITE))
