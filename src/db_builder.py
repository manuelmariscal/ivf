import sqlite3
from pathlib import Path
import polars as pl
from utils import Logger

DEFAULT_CSV = Path('data/E2_develop.csv')
DEFAULT_DB = Path('data/e2.db')

COLUMN_MAP = {
    'IDENTIFICADOR': 'patientId',
    'Stimulation day': 'stimDay',
    'E2': 'e2',
    'monitoring_visit': 'visit',
    'Follicle 6-7mm': 'follicle6_7',
    'Follicle 8-9mm': 'follicle8_9',
    'Follicle 10-11mm': 'follicle10_11',
    'Follicle 12-13mm': 'follicle12_13',
    'Follicle 14-15mm': 'follicle14_15',
    'Follicle 16-17mm': 'follicle16_17',
    'Follicle 18-19mm': 'follicle18_19',
    'Follicle 20-21mm': 'follicle20_21',
    'Follicle 22-23': 'follicle22_23',
    'Follicle >23mm': 'follicleGt23',
    'Follicle <7mm': 'follicleLt7',
    'Follicle 8-11mm': 'follicle8_11',
    'Follicle 12-17mm': 'follicle12_17',
    'Follicle 18-21mm': 'follicle18_21',
    'Follicle >22mm': 'follicleGt22',
    'Age': 'age',
}

def build_database(csv_path: Path = DEFAULT_CSV, db_path: Path = DEFAULT_DB) -> None:
    Logger.info(f'Loading CSV {csv_path}')
    df = pl.read_csv(csv_path)
    df = df.rename(COLUMN_MAP)
    Logger.info(f'Saving to SQLite {db_path}')
    conn = sqlite3.connect(db_path)
    # polars does not write directly to sqlite without optional features.
    # Convert to pandas for compatibility.
    df.to_pandas().to_sql('measurements', conn, if_exists='replace', index=False)
    conn.close()
    Logger.success('Database created')

if __name__ == '__main__':
    build_database()
