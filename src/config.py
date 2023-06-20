import pathlib
import string

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

ENDS = "[ENDS]"
DEFAULT_CHARS = list(string.ascii_lowercase) + [ENDS]
