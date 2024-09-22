from app.instructor_hub import InstructorHub

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