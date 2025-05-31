import json
import glob
import os


def load_qa():
    # Get all QA JSON files from the generated_qa directory
    qa_files = glob.glob('generated_qa/*.json')

    # Initialize lists to store all questions and answers
    all_questions = []
    all_answers = []

    # Load data from each file
    for file_path in qa_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if the loaded data is a list
            if isinstance(data, list):
                # Iterate through the list of Q&A pairs
                for qa_pair in data:
                    # Ensure each item in the list is a dictionary with 'question' and 'answer' keys
                    if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answer' in qa_pair:
                        all_questions.append(qa_pair['question'])
                        all_answers.append(qa_pair['answer'])
                    else:
                        print(f"Warning: Skipping invalid item in {file_path}: {qa_pair}")
            else:
                 print(f"Warning: Data in {file_path} is not a list. Skipping file.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

        # return questions and answers
        return all_questions, all_answers
