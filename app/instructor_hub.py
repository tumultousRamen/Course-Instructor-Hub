import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Union

from data_loader import DataLoader
from query_processor import QueryProcessor
from config import Config

class InstructorHub:
    def __init__(self):
        self._initialize_nltk()
        self.data = DataLoader.load_all_data()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.query_processor = QueryProcessor(self.stop_words, self.get_predefined_queries())

    @staticmethod
    def _initialize_nltk():
        nltk.data.path.append(str(Config.NLTK_DATA_PATH))
        needed_corpora = ['punkt', 'stopwords', 'vader_lexicon']
        for corpus in needed_corpora:
            try:
                nltk.data.find(f'corpora/{corpus}')
            except LookupError:
                nltk.download(corpus, download_dir=Config.NLTK_DATA_PATH, quiet=True)

    def get_predefined_queries(self) -> Dict[str, callable]:
        return {
            "enrolled students": self.enrolled_students,
            "completed module": self.completed_module,
            "completed assessment": self.completed_assessment,
            "completed course": self.completed_course,
            "average score": self.average_score,
            "highest completion rate": self.highest_completion_rate,
            "top performing students": self.top_performing_students,
            "overall completion rate": self.overall_completion_rate,
            "student performance": self.student_performance,
            "final exam performance": self.final_exam_performance,
            "assessment difficulty": self.assessment_difficulty,
            "module feedback": self.module_feedback,
            "overall course sentiment": self.overall_course_sentiment
        }

    def process_query(self, query: str) -> str:
        return self.query_processor.process_query(query)

    def enrolled_students(self, query: str) -> str:
        return f"Total enrolled students: {len(self.data['students'])}"

    def completed_module(self, query: str) -> str:
        module = self.extract_module_or_assessment(query, self.data['modules'], 'module')
        if not module:
            module = input("Which module (enter ID or name)? ")
        
        module_id = self.get_module_id(module)
        if module_id is None:
            return f"Module '{module}' not found."
        
        completed = self.data['student_module_completions'][
            self.data['student_module_completions']['module_id'] == module_id
        ]['student_id'].nunique()
        module_name = self.data['modules'][self.data['modules']['module_id'] == module_id]['module_name'].iloc[0]
        return f"Students completed {module_name}: {completed}"

    def completed_assessment(self, query: str) -> str:
        assessment = self.extract_module_or_assessment(query, self.data['assessments'], 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        completed = self.data['student_assessment_completions'][
            self.data['student_assessment_completions']['assessment_id'] == assessment_id
        ]['student_id'].nunique()
        assessment_name = self.data['assessments'][
            self.data['assessments']['assessment_id'] == assessment_id
        ]['assessment_name'].iloc[0]
        return f"Students completed {assessment_name}: {completed}"

    def completed_course(self, query: str) -> str:
        completed = self.data['students'][
            self.data['students']['student_id'].isin(
                self.data['student_module_completions'].groupby('student_id').filter(
                    lambda x: len(x) == len(self.data['modules'])
                )['student_id']
            )
        ]['student_id'].nunique()
        return f"Students completed the course: {completed}"

    def average_score(self, query: str) -> str:
        assessment = self.extract_module_or_assessment(query, self.data['assessments'], 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        avg_score = self.data['student_assessment_completions'][
            self.data['student_assessment_completions']['assessment_id'] == assessment_id
        ]['score'].mean()
        assessment_name = self.data['assessments'][
            self.data['assessments']['assessment_id'] == assessment_id
        ]['assessment_name'].iloc[0]
        return f"Average score for {assessment_name}: {avg_score:.2f}"

    def highest_completion_rate(self, query: str) -> str:
        completion_rates = self.data['student_module_completions'].groupby('module_id')['student_id'].nunique() / len(self.data['students'])
        highest_rate_module = self.data['modules'].loc[completion_rates.idxmax(), 'module_name']
        return f"Module with highest completion rate: {highest_rate_module}"

    def top_performing_students(self, query: str) -> str:
        top_students = self.data['student_assessment_completions'].groupby('student_id')['score'].mean().nlargest(5)
        top_students_names = self.data['students'][self.data['students']['student_id'].isin(top_students.index)]['name']
        return f"Top performing students: {', '.join(top_students_names)}"

    def overall_completion_rate(self, query: str) -> str:
        completion_rate = (
            self.data['students']['student_id'].isin(
                self.data['student_module_completions'].groupby('student_id').filter(
                    lambda x: len(x) == len(self.data['modules'])
                )['student_id']
            ).sum() / len(self.data['students'])
        ) * 100
        return f"Overall course completion rate: {completion_rate:.2f}%"

    def student_performance(self, query: str) -> str:
        student = self.extract_student(query)
        if not student:
            student = input("Which student (enter ID or name)? ")
        
        student_id = self.get_student_id(student)
        if student_id is None:
            return f"Student '{student}' not found."
        
        assessments = self.data['student_assessment_completions'][
            self.data['student_assessment_completions']['student_id'] == student_id
        ]
        avg_score = assessments['score'].mean()
        completed_modules = self.data['student_module_completions'][
            self.data['student_module_completions']['student_id'] == student_id
        ]['module_id'].nunique()
        student_name = self.data['students'][self.data['students']['student_id'] == student_id]['name'].iloc[0]
        
        return f"Performance for {student_name}:\nAverage Assessment Score: {avg_score:.2f}\nCompleted Modules: {completed_modules}/{len(self.data['modules'])}"

    def final_exam_performance(self, query: str) -> str:
        final_exam = self.data['assessments'][self.data['assessments']['assessment_name'].str.contains('Final Exam', case=False)]
        if final_exam.empty:
            return "No Final Exam found in the assessments."
        
        final_exam_id = final_exam['assessment_id'].iloc[0]
        final_exam_scores = self.data['student_assessment_completions'][
            self.data['student_assessment_completions']['assessment_id'] == final_exam_id
        ]
        
        avg_score = final_exam_scores['score'].mean()
        max_score = final_exam_scores['score'].max()
        min_score = final_exam_scores['score'].min()
        
        return f"Final Exam Performance:\nAverage Score: {avg_score:.2f}\nHighest Score: {max_score}\nLowest Score: {min_score}"

    def assessment_difficulty(self, query: str) -> str:
        assessment = self.extract_module_or_assessment(query, self.data['assessments'], 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        assessment_scores = self.data['student_assessment_completions'][
            self.data['student_assessment_completions']['assessment_id'] == assessment_id
        ]
        avg_score = assessment_scores['score'].mean()
        avg_attempts = assessment_scores['attempts'].mean()
        
        difficulty = "Easy" if avg_score > 80 else "Moderate" if avg_score > 60 else "Difficult"
        
        assessment_name = self.data['assessments'][
            self.data['assessments']['assessment_id'] == assessment_id
        ]['assessment_name'].iloc[0]
        return f"Difficulty of {assessment_name}:\nAverage Score: {avg_score:.2f}\nAverage Attempts: {avg_attempts:.2f}\nDifficulty Level: {difficulty}"

    def module_feedback(self, query: str) -> str:
        module = self.extract_module_or_assessment(query, self.data['modules'], 'module')
        if not module:
            module = input("Which module (enter ID or name)? ")
        
        module_id = self.get_module_id(module)
        if module_id is None:
            return f"Module '{module}' not found."
        
        module_feedback = self.data['student_module_completions'][
            self.data['student_module_completions']['module_id'] == module_id
        ]
        avg_rating = module_feedback['rating'].mean()
        
        feedback_sentiments = module_feedback['feedback'].apply(
            lambda x: self.sia.polarity_scores(x)['compound'] if pd.notna(x) else None
        )
        avg_sentiment = feedback_sentiments.mean()
        
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
        module_name = self.data['modules'][self.data['modules']['module_id'] == module_id]['module_name'].iloc[0]
        return f"Feedback for {module_name}:\nAverage Rating: {avg_rating:.2f}/5\nOverall Sentiment: {sentiment_label} ({avg_sentiment:.2f})"

    def overall_course_sentiment(self, query: str) -> str:
        all_feedback = self.data['student_module_completions']['feedback'].dropna()
        sentiments = all_feedback.apply(lambda x: self.sia.polarity_scores(x)['compound'])
        avg_sentiment = sentiments.mean()
        
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
        positive_comments = sum(sentiments > 0.05)
        negative_comments = sum(sentiments < -0.05)
        neutral_comments = sum((sentiments >= -0.05) & (sentiments <= 0.05))
        
        return f"Overall Course Sentiment:\nAverage Sentiment: {sentiment_label} ({avg_sentiment:.2f})\nPositive Comments: {positive_comments}\nNeutral Comments: {neutral_comments}\nNegative Comments: {negative_comments}"

    def extract_module_or_assessment(self, query: str, df: pd.DataFrame, item_type: str) -> Optional[str]:
        for item in df[f'{item_type}_name']:
            if item.lower() in query.lower():
                return item
        for item in df[f'{item_type}_id'].astype(str):
            if item in query:
                return item
        return None

    def get_module_id(self, module: Union[str, int]) -> Optional[int]:
        if isinstance(module, int) or (isinstance(module, str) and module.isdigit()):
            module_id = int(module)
            if module_id in self.data['modules']['module_id'].values:
                return module_id
        else:
            module_row = self.data['modules'][self.data['modules']['module_name'].str.lower() == str(module).lower()]
            if not module_row.empty:
                return module_row['module_id'].iloc[0]
        return None

    def get_assessment_id(self, assessment: Union[str, int]) -> Optional[int]:
        if isinstance(assessment, int) or (isinstance(assessment, str) and assessment.isdigit()):
            assessment_id = int(assessment)
            if assessment_id in self.data['assessments']['assessment_id'].values:
                return assessment_id
        else:
            assessment_row = self.data['assessments'][
                self.data['assessments']['assessment_name'].str.lower() == str(assessment).lower()
            ]
            if not assessment_row.empty:
                return assessment_row['assessment_id'].iloc[0]
        return None

    def extract_student(self, query: str) -> Optional[str]:
        for name in self.data['students']['name']:
            if name.lower() in query.lower():
                return name
        for student_id in self.data['students']['student_id'].astype(str):
            if student_id in query:
                return student_id
        return None

    def get_student_id(self, student: Union[str, int]) -> Optional[int]:
        if isinstance(student, int) or (isinstance(student, str) and student.isdigit()):
            student_id = int(student)
            if student_id in self.data['students']['student_id'].values:
                return student_id
        else:
            student_row = self.data['students'][self.data['students']['name'].str.lower() == str(student).lower()]
            if not student_row.empty:
                return student_row['student_id'].iloc[0]
        return None