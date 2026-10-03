"""Set up the local environment and open Portfolio Lab in a browser."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 14):
        print("Portfolio Lab needs Python 3.14. Install it from https://www.python.org/downloads/")
        return 1

    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = ROOT / "requirements.txt"
    stamp = environment / "portfolio-lab-requirements.sha256"
    try:
        print("Portfolio Lab - starting your local dashboard", flush=True)
        if not python.exists():
            print("Creating the app's Python environment...", flush=True)
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        version = subprocess.run([str(python), "-c", "import sys; print(sys.version_info[:2])"],
                                 check=True, capture_output=True, text=True)
        if version.stdout.strip() != "(3, 14)":
            print("The .venv folder uses another Python version. Remove only .venv, then try again.")
            return 1

        fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
        if not stamp.exists() or stamp.read_text() != fingerprint:
            print("Installing the app's packages. First setup needs internet and may take a few minutes...",
                  flush=True)
            subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                            "-r", str(requirements)], check=True)
            stamp.write_text(fingerprint)

        print("Opening Portfolio Lab in your browser. Keep this window open while using it.", flush=True)
        print("If no tab opens, use the Local URL printed below. Press Ctrl+C here to stop.", flush=True)
        subprocess.run([str(python), "-m", "streamlit", "run", "app.py",
                        "--server.headless=false", "--server.showEmailPrompt=false",
                        *sys.argv[1:]], cwd=ROOT, check=True)
        return 0
    except KeyboardInterrupt:
        print("\nPortfolio Lab stopped.")
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"\nPortfolio Lab could not start: {error}")
        print("For a download failure, reconnect to the internet and run the launcher again.")
        print("For other errors, see 'Troubleshooting' in README.md; keep the message above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
