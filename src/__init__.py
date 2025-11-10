import sys
from pathlib import Path

# Determine ROOT path of /src directory
SRC_ROOT = Path(__file__).parent.resolve()

# Add /src path to sys.path if it doesn't exist yet
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
    print(f"Added {SRC_ROOT} to sys.path")
