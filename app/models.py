from dataclasses import dataclass
from typing import Optional

@dataclass
class Student:
    id: int
    name: str

@dataclass
class Module:
    id: int
    name: str

@dataclass
class Assessment:
    id: int
    name: str

@dataclass
class ModuleCompletion:
    student_id: int
    module_id: int
    minutes_spent: int
    feedback: Optional[str]
    rating: int

@dataclass
class AssessmentCompletion:
    student_id: int
    assessment_id: int
    score: float
    attempts: int