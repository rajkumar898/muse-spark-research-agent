"""Smoke tests for the terminal version (cli.py). Mock model, no network."""
import os
import subprocess
import sys
from pathlib import Path

from pdf_helper import make_text_pdf

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, stdin=""):
    env = {**os.environ, "MINI_MUSE_MOCK": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args], input=stdin, capture_output=True,
                          text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=60)


def test_cli_calculation():
    out = run_cli("Calculate the percentage increase from 100 to 125.")
    assert out.returncode == 0
    assert "MOCK MODE" in out.stdout and "Calculated pct_change(100, 125) = 25.0" in out.stdout


def test_cli_pdf_excerpt_is_shown(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_text_pdf(["This study uses the NREL solar irradiance dataset and RMSE metrics."]))
    out = run_cli("--mode", "tool", "--pdf", str(pdf), "What dataset does this paper use?")
    assert out.returncode == 0 and "NREL solar irradiance dataset" in out.stdout


def test_cli_rejects_missing_pdf():
    out = run_cli("--pdf", "does-not-exist.pdf", "question here")
    assert out.returncode != 0 and "Cannot use that PDF" in out.stderr
