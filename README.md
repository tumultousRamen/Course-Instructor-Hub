# Instructor Hub

This project implements an AI-powered Instructor Hub for course management and analysis. It leverages Natural Language Processing (NLP) to interpret instructor queries and provides meaningful insights about course performance, student engagement, and content quality. 

## Overview
The application is a command-line tool that allows instructors to ask relevant questions about the course they are delivering in natural language. The application uses Natural Language Toolkit (NLTK) for text processing and sentiment analysis, pandas for data manipulation, and a custom query processing system to match user queries to predefined functions that fetch relevant insights from provided data. 

## Scope of Prompts
The application can answer queries related to:
### Enrollment:
1. Total number of students enrolled

### Engagement:
1. Number of students who completed a specific module (can specifed with either a module ID or name in the base prompt -- if module information is not provided or doesn't the data, the application will ask the instructor to clarify!)
2. Number of students who completed a specific assessment (same as above)
3. Number of students who completed the course
4. Module with the highest completion rate
5. Overall course completion rate

### Performance
1. Average score for a specific assessment
2. Top performing students
3. Individual student performance
4. Final exam performance statistics
5. Assessment difficulty analysis

### Content and Quality 
1. Feedback and sentiment analysis for specific modules
2. Overall course sentiment analysis


## Setup

1. Clone the repository
2. Change directory to app
    ```
    cd app
    ```
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Ensure your data files are in the `data/project/` directory

## Running the Application

Run the main script:

```
python main.py
```

Follow the prompts to interact with the Instructor Hub.

## How it Works

1. The application loads course data from CSV files using pandas.
2. It initializes NLTK for natural language processing and sentiment analysis.
3. When a user inputs a query, the system tokenizes and processes it using NLTK.
4. The processed query is matched against predefined query patterns.
5. If a match is found, the corresponding function is called to analyze the data and generate a response.
6. The response is returned to the user in natural language.

The system uses a flexible query matching system that allows for variations in how questions are asked, making the interaction more natural and user-friendly.

## Improvements and Expansion
The projects delivers a basic application that can answer a range of queries concerning a course. Some avenues of improvement could be:
### Natural Language Understanding:
1. Implement more advanced NLU techniques, such as intent classification and entity recognition, to better understand complex queries.
2. Use machine learning models (e.g., BERT or GPT) for more accurate query interpretation.

### Data Visualization
1. Integrate data visualization libraries like Matplotlib or Plotly to generate charts and graphs for visual representation of data.

### Predictive Analysis
1. Implement machine learning models to predict student performance or identify at-risk students based on their engagement and assessment data.
2. Develop trend analysis features to track course performance over time.

### Custom Course Metrics
1. Allow instructors to define custom metrics or Key Performance Indicators (KPIs) for their courses.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details