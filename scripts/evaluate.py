import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import json
import gensim
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from pcst_fast import pcst_fast
import re
from datetime import datetime
from load_qa_data import load_qa

def evaluate_response(question, context, response, label=None, report=False):
    """
    Evaluates the LLM's response based on clarity, exactitude, and context adherence.

    Args:
        question (str): Original question
        context (str): Retrieved knowledge graph context
        response (str): LLM's response to evaluate
        label (str, optional): Ground truth label for comparison
        report (bool): Whether to save evaluation to file

    Returns:
        dict: Evaluation scores and feedback
    """     
    
    if report:
        current_date = datetime.now().strftime("%Y-%m-%d")
        with open(f"evaluation_{current_date}.txt", "a") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Label: {label}\n")
        

    evaluation_prompt = f"""You are an expert evaluator of medical knowledge responses. Evaluate the following response based on three criteria:

1. Clarity (0-10): How clear and well-structured is the response? 0 is the worst, 1 is the best.
2. Exactitude (0-10): How accurate and precise is the information provided? 0 is the worst, 1 is the best.
3. Context Adherence (0-10): How well does the response stick to the provided knowledge graphs? 0 is the worst, 1 is the best.
4. Relevance (0-10): How relevant is the retrieved Knowledge Graph Context to the question? 0 is the worst, 1 is the best.
5. Completeness (0-10): How complete and thorough is the response? 0 is the worst, 1 is the best.
6. Logical Flow (0-10): How coherent and well-structured is the response? 0 is the worst, 1 is the best.
7. Uncertainty Handling (0-10): How well does the response acknowledge limitations and uncertainties? 0 is the worst, 1 is the best.


Question: {question}

Knowledge Graph Context:
{context}

Response to Evaluate:
{response}

Provide your evaluation in the following format:
CLARITY: [score]/10 - [brief explanation]
EXACTITUDE: [score]/10 - [brief explanation]
CONTEXT ADHERENCE: [score]/10 - [brief explanation]
RELEVANCE: [score]/10 - [brief explanation]
COMPLETENESS: [score]/10 - [brief explanation]
LOGICAL FLOW: [score]/10 - [brief explanation]
UNCERTAINTY HANDLING: [score]/10 - [brief explanation]
OVERALL FEEDBACK: [average score] and 2-3 sentences summarizing the evaluation]
"""

    evaluation = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of medical knowledge responses."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent evaluation
        top_p=0.95,
        max_tokens=1024,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )

    print("\n=== Response Evaluation ===\n")
    
    # Initialize variables to store the complete response
    full_evaluation = ""
    scores = {
        "clarity_score": None,
        "exactitude_score": None,
        "context_adherence_score": None,
        "relevance_score": None,
        "completeness_score": None,
        "logical_flow_score": None,
        "uncertainty_handling_score": None,
        "overall_feedback": None
    }
    
    for chunk in evaluation:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="")
            full_evaluation += content

    # Extract scores from the complete evaluation
    scores["clarity_score"] = re.findall(r"CLARITY: (\d+(?:\.\d+)?)", full_evaluation)
    scores["exactitude_score"] = re.findall(r"EXACTITUDE: (\d+(?:\.\d+)?)", full_evaluation)
    scores["context_adherence_score"] = re.findall(r"CONTEXT ADHERENCE: (\d+(?:\.\d+)?)", full_evaluation)
    scores["relevance_score"] = re.findall(r"RELEVANCE: (\d+(?:\.\d+)?)", full_evaluation)
    scores["completeness_score"] = re.findall(r"COMPLETENESS: (\d+(?:\.\d+)?)", full_evaluation)
    scores["logical_flow_score"] = re.findall(r"LOGICAL FLOW: (\d+(?:\.\d+)?)", full_evaluation)
    scores["uncertainty_handling_score"] = re.findall(r"UNCERTAINTY HANDLING: (\d+(?:\.\d+)?)", full_evaluation)
    scores["overall_feedback"] = re.findall(r"OVERALL FEEDBACK: (.*?)(?=\n|$)", full_evaluation)

    # Convert list matches to single values
    for key in scores:
        if scores[key]:
            scores[key] = scores[key][0] if isinstance(scores[key], list) else scores[key]

    print("\n=== Evaluation Complete ===\n")
    
    if report:
        with open(f"evaluation_{current_date}.txt", "a") as f:
            f.write("\nScores:\n")
            f.write(json.dumps(scores, indent=2))
            f.write("\n\n" + "="*50 + "\n\n")
    
    return scores


# # Example usage:
# for question in questions:
#     print(f"\n\nQuestion: {question}\n")
#     # Get the response
#     full_response = generate_response(question)
#     print(f"Response: {full_response}\n")
#     sub_graphs, descriptions = retreival(question, k=3)
#     context = "\n".join(descriptions)  # Combine all knowledge graph descriptions
#     evaluate_response(question, context, full_response)
