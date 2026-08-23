import sys
import runpy
from pathlib import Path

# Bulletproof Repository Root Finder for Streamlit Cloud & Local
current = Path(__file__).resolve().parent
repo_root = current
while repo_root != repo_root.parent:
    if (repo_root / "src").is_dir() and (repo_root / "src" / "config.py").is_file():
        break
    repo_root = repo_root.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Run main UI script
ui_script = repo_root / "src" / "ui" / "app.py"
if ui_script.exists():
    runpy.run_path(str(ui_script), run_name="__main__")