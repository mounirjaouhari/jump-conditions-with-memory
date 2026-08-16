"""fading_memory — computation core of the article "Jump conditions with memory".

Single entry point for all the physics and the provenance tracking.
No script may reimplement a formula present here.

    from fading_memory import load_config, RunRecorder
    import fading_memory.physics as ph
"""

import sys

__version__ = "0.1.0"


def _console_utf8() -> None:
    """The Windows console uses cp1252: accents and symbols crash there.

    Without this, a simple `print("✓ done")` raises UnicodeEncodeError on
    Windows and nowhere else — a project that only runs on its author's
    machine is not reproducible.
    """
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # redirected / non-reconfigurable stream
            pass


_console_utf8()

from .config import Config, load_config  # noqa: E402
from .provenance import RunRecorder  # noqa: E402

__all__ = ["Config", "load_config", "RunRecorder", "__version__"]
