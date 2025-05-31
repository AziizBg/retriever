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
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_questions.extend(data['questions'])
            all_answers.extend(data['answers'])

    print(f"Total number of QA pairs loaded: {len(all_questions)}")
    print("\nSample questions:")
    for i, q in enumerate(all_questions[:5]):
        print(f"{i+1}. {q}")        