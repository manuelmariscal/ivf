"""Data loading utilities."""

import sqlite3
from pathlib import Path
import polars as pl
from utils import Logger

class DataLoaderError(Exception):
    pass

class RealDataLoader:
    """Load measurements from a SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def load(self) -> pl.DataFrame:
        if not self.db_path.exists():
            raise DataLoaderError(f"Database {self.db_path} not found")
        Logger.info(f"Loading data from {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        df = pl.read_database("SELECT * FROM measurements", conn)
        conn.close()
        Logger.success(f"Loaded {len(df)} rows")
        return df
