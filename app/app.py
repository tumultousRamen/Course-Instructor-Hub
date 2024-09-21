import csv
import os
from collections import defaultdict
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

download_nltk_data()

class InstructorHub:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'project'))
        
        self.students = pd.read_csv(os.path.join(data_dir, 'students.csv'))
        self.modules = pd.read_csv(os.path.join(data_dir, 'modules.csv'))
        self.assessments = pd.read_csv(os.path.join(data_dir, 'assessments.csv'))
        self.student_module_completions = pd.read_csv(os.path.join(data_dir, 'student_module_completions.csv'))
        self.student_assessment_completions = pd.read_csv(os.path.join(data_dir, 'student_assessment_completions.csv'))

        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.predefined_queries = self.get_predefined_queries()

    def get_predefined_queries(self):
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

    def process_query(self, query):
        tokens = word_tokenize(query.lower())
        tokens = [w for w in tokens if w not in self.stop_words]
        
        # Define more specific keywords for each query type
        query_keywords = {
            "enrolled students": ["enrolled", "students", "total"],
            "completed module": ["completed", "module"],
            "completed assessment": ["completed", "assessment"],
            "completed course": ["completed", "course", "entire"],
            "average score": ["average", "score"],
            "highest completion rate": ["highest", "completion", "rate"],
            "top performing students": ["top", "performing", "students"],
            "overall completion rate": ["overall", "completion", "rate"],
            "student performance": ["student", "performance", "doing"],
            "final exam performance": ["final", "exam"],
            "assessment difficulty": ["difficult", "difficulty", "assessment", "quiz"],
            "module feedback": ["think", "feedback", "module"],
            "overall course sentiment": ["overall", "sentiment", "course"]
        }
        
        best_match = None
        max_overlap = 0
        
        for key, keywords in query_keywords.items():
            overlap = len(set(tokens) & set(keywords))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = key
        
        if best_match and max_overlap > 0:
            return self.predefined_queries[best_match](query)
        else:
            return "I'm sorry, I don't understand that query. Could you please rephrase it?"

    def enrolled_students(self, query):
        return f"Total enrolled students: {len(self.students)}"

    def completed_module(self, query):
        module = self.extract_module_or_assessment(query, self.modules, 'module')
        if not module:
            module = input("Which module (enter ID or name)? ")
        
        module_id = self.get_module_id(module)
        if module_id is None:
            return f"Module '{module}' not found."
        
        completed = self.student_module_completions[self.student_module_completions['module_id'] == module_id]['student_id'].nunique()
        module_name = self.modules[self.modules['module_id'] == module_id]['module_name'].iloc[0]
        return f"Students completed {module_name}: {completed}"

    def completed_assessment(self, query):
        assessment = self.extract_module_or_assessment(query, self.assessments, 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        completed = self.student_assessment_completions[self.student_assessment_completions['assessment_id'] == assessment_id]['student_id'].nunique()
        assessment_name = self.assessments[self.assessments['assessment_id'] == assessment_id]['assessment_name'].iloc[0]
        return f"Students completed {assessment_name}: {completed}"

    def completed_course(self, query):
        completed = self.students[self.students['student_id'].isin(self.student_module_completions.groupby('student_id').filter(lambda x: len(x) == len(self.modules))['student_id'])]['student_id'].nunique()
        return f"Students completed the course: {completed}"

    def average_score(self, query):
        assessment = self.extract_module_or_assessment(query, self.assessments, 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        avg_score = self.student_assessment_completions[self.student_assessment_completions['assessment_id'] == assessment_id]['score'].mean()
        assessment_name = self.assessments[self.assessments['assessment_id'] == assessment_id]['assessment_name'].iloc[0]
        return f"Average score for {assessment_name}: {avg_score:.2f}"

    def highest_completion_rate(self, query):
        completion_rates = self.student_module_completions.groupby('module_id')['student_id'].nunique() / len(self.students)
        highest_rate_module = self.modules.loc[completion_rates.idxmax(), 'module_name']
        return f"Module with highest completion rate: {highest_rate_module}"

    def top_performing_students(self, query):
        top_students = self.student_assessment_completions.groupby('student_id')['score'].mean().nlargest(5)
        top_students_names = self.students[self.students['student_id'].isin(top_students.index)]['name']
        return f"Top performing students: {', '.join(top_students_names)}"

    def overall_completion_rate(self, query):
        completion_rate = (self.students['student_id'].isin(self.student_module_completions.groupby('student_id').filter(lambda x: len(x) == len(self.modules))['student_id']).sum() / len(self.students)) * 100
        return f"Overall course completion rate: {completion_rate:.2f}%"

    def extract_module_or_assessment(self, query, df, item_type):
        for item in df[f'{item_type}_name']:
            if item.lower() in query.lower():
                return item
        for item in df[f'{item_type}_id'].astype(str):
            if item in query:
                return item
        return None

    def get_module_id(self, module):
        if module.isdigit():
            if int(module) in self.modules['module_id'].values:
                return int(module)
        else:
            module_row = self.modules[self.modules['module_name'].str.lower() == module.lower()]
            if not module_row.empty:
                return module_row['module_id'].iloc[0]
        return None

    def get_assessment_id(self, assessment):
        if assessment.isdigit():
            if int(assessment) in self.assessments['assessment_id'].values:
                return int(assessment)
        else:
            assessment_row = self.assessments[self.assessments['assessment_name'].str.lower() == assessment.lower()]
            if not assessment_row.empty:
                return assessment_row['assessment_id'].iloc[0]
        return None
    
    def student_performance(self, query):
        student = self.extract_student(query)
        if not student:
            student = input("Which student (enter ID or name)? ")
        
        student_id = self.get_student_id(student)
        if student_id is None:
            return f"Student '{student}' not found."
        
        assessments = self.student_assessment_completions[self.student_assessment_completions['student_id'] == student_id]
        avg_score = assessments['score'].mean()
        completed_modules = self.student_module_completions[self.student_module_completions['student_id'] == student_id]['module_id'].nunique()
        student_name = self.students[self.students['student_id'] == student_id]['name'].iloc[0]
        
        return f"Performance for {student_name}:\nAverage Assessment Score: {avg_score:.2f}\nCompleted Modules: {completed_modules}/{len(self.modules)}"


    def final_exam_performance(self, query):
        final_exam = self.assessments[self.assessments['assessment_name'].str.contains('Final Exam', case=False)]
        if final_exam.empty:
            return "No Final Exam found in the assessments."
        
        final_exam_id = final_exam['assessment_id'].iloc[0]
        final_exam_scores = self.student_assessment_completions[self.student_assessment_completions['assessment_id'] == final_exam_id]
        
        avg_score = final_exam_scores['score'].mean()
        max_score = final_exam_scores['score'].max()
        min_score = final_exam_scores['score'].min()
        
        return f"Final Exam Performance:\nAverage Score: {avg_score:.2f}\nHighest Score: {max_score}\nLowest Score: {min_score}"

    def assessment_difficulty(self, query):
        assessment = self.extract_module_or_assessment(query, self.assessments, 'assessment')
        if not assessment:
            assessment = input("Which assessment (enter ID or name)? ")
        
        assessment_id = self.get_assessment_id(assessment)
        if assessment_id is None:
            return f"Assessment '{assessment}' not found."
        
        assessment_scores = self.student_assessment_completions[self.student_assessment_completions['assessment_id'] == assessment_id]
        avg_score = assessment_scores['score'].mean()
        avg_attempts = assessment_scores['attempts'].mean()
        
        difficulty = "Easy" if avg_score > 80 else "Moderate" if avg_score > 60 else "Difficult"
        
        assessment_name = self.assessments[self.assessments['assessment_id'] == assessment_id]['assessment_name'].iloc[0]
        return f"Difficulty of {assessment_name}:\nAverage Score: {avg_score:.2f}\nAverage Attempts: {avg_attempts:.2f}\nDifficulty Level: {difficulty}"



    def module_feedback(self, query):
        module = self.extract_module_or_assessment(query, self.modules, 'module')
        if not module:
            module = input("Which module (enter ID or name)? ")
        
        module_id = self.get_module_id(module)
        if module_id is None:
            return f"Module '{module}' not found."
        
        module_feedback = self.student_module_completions[self.student_module_completions['module_id'] == module_id]
        avg_rating = module_feedback['rating'].mean()
        
        feedback_sentiments = module_feedback['feedback'].apply(lambda x: self.sia.polarity_scores(x)['compound'] if pd.notna(x) else None)
        avg_sentiment = feedback_sentiments.mean()
        
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
        module_name = self.modules[self.modules['module_id'] == module_id]['module_name'].iloc[0]
        return f"Feedback for {module_name}:\nAverage Rating: {avg_rating:.2f}/5\nOverall Sentiment: {sentiment_label} ({avg_sentiment:.2f})"

    def overall_course_sentiment(self, query):
        all_feedback = self.student_module_completions['feedback'].dropna()
        sentiments = all_feedback.apply(lambda x: self.sia.polarity_scores(x)['compound'])
        avg_sentiment = sentiments.mean()
        
        sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
        positive_comments = sum(sentiments > 0.05)
        negative_comments = sum(sentiments < -0.05)
        neutral_comments = sum((sentiments >= -0.05) & (sentiments <= 0.05))
        
        return f"Overall Course Sentiment:\nAverage Sentiment: {sentiment_label} ({avg_sentiment:.2f})\nPositive Comments: {positive_comments}\nNeutral Comments: {neutral_comments}\nNegative Comments: {negative_comments}"

    def extract_student(self, query):
        for name in self.students['name']:
            if name.lower() in query.lower():
                return name
        for student_id in self.students['student_id'].astype(str):
            if student_id in query:
                return student_id
        return None

    def get_student_id(self, student):
        if student.isdigit():
            if int(student) in self.students['student_id'].values:
                return int(student)
        else:
            student_row = self.students[self.students['name'].str.lower() == student.lower()]
            if not student_row.empty:
                return student_row['student_id'].iloc[0]
        return None


def main():
    try:
        hub = InstructorHub()
        
        print("Welcome to the Enhanced AI Instructor Hub!")
        print("You can ask questions about student enrollment, module completion, assessment performance, content quality, and more.")
        print("You can include specific module, assessment, or student names/IDs in your query.")
        print("Type 'exit' to quit the program.")

        while True:
            query = input("\nWhat would you like to know? ")
            if query.lower() == 'exit':
                print("Thank you for using the Enhanced AI Instructor Hub. Goodbye!")
                break
            response = hub.process_query(query)
            print(response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If this is related to NLTK data, please try running the script again.")

if __name__ == "__main__":
    main()