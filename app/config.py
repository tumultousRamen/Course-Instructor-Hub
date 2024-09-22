import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'project'
    NLTK_DATA_PATH = BASE_DIR / 'nltk_data'

    @classmethod
    def get_csv_path(cls, filename: str) -> Path:
        return cls.DATA_DIR / filename