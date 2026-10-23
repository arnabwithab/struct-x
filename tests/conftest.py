"""Make src/ importable without installing the package (system python has torch)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
