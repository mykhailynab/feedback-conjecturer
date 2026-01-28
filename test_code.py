import polars as pl
from pathlib import Path
import json
import csv

LOG_DIR = Path("/kaggle/working/aimo3_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

ATTEMPTS_PATH = LOG_DIR / "attempts.jsonl"
SOLUTIONS_PATH = LOG_DIR / "solutions.csv"

# Load reference.csv for local runs (and also as a GT fallback)
REFERENCE_PATH = "reference.csv"
try:
    _ref_df = pl.read_csv(REFERENCE_PATH)
    # Expect columns: id, question, answer (per your description)
    REF_ANSWER_BY_ID = dict(zip(_ref_df["id"].to_list(), _ref_df["answer"].to_list()))
    REF_QUESTION_BY_ID = dict(zip(_ref_df["id"].to_list(), _ref_df["question"].to_list()))
    TOTAL_PROBLEMS = _ref_df.height
except Exception:
    REF_ANSWER_BY_ID = {}
    REF_QUESTION_BY_ID = {}
    TOTAL_PROBLEMS = 50  # fallback
