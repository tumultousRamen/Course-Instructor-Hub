import pandas as pd
from pathlib import Path
from typing import Dict

from config import Config

class DataLoader:
    @staticmethod
    def load_csv(filename: str) -> pd.DataFrame:
        file_path = Config.get_csv_path(filename)
        return pd.read_csv(file_path)

    @classmethod
    def load_all_data(cls) -> Dict[str, pd.DataFrame]:
        return {
            'students': cls.load_csv('students.csv'),
            'modules': cls.load_csv('modules.csv'),
            'assessments': cls.load_csv('assessments.csv'),
            'student_module_completions': cls.load_csv('student_module_completions.csv'),
            'student_assessment_completions': cls.load_csv('student_assessment_completions.csv'),
        }