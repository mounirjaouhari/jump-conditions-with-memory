"""Make `fading_memory` importable by pytest without prior installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
