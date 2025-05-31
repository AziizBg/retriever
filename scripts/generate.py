def construct_rag_prompt(question, descriptions):
    """
    Constructs a prompt for RAG using the retrieved knowledge graph descriptions.
    """
    # Start with system message
    system_message = """You are a medical knowledge assistant that helps answer questions based on medical knowledge graphs.
Use only the provided knowledge graphs to answer the question. If the knowledge graphs don't contain enough information to fully answer the question,
say I don't know. Be precise and factual in your responses. Don't cite the node and graph ids in your responses for example (Knowledge Graph number). Format the responses for more clarity.
"""

    # Format the context from knowledge graphs
    context = "Here are the relevant medical knowledge graphs:\n\n"
    for i, desc in enumerate(descriptions, 1):
        # Split the description into nodes and edges
        parts = desc.split('\n\n')
        nodes = parts[0]
        edges = parts[1] if len(parts) > 1 else ""

        context += f"Knowledge Graph {i}:\n"
        context += f"Nodes:\n{nodes}\n"
        if edges:
            context += f"Relationships:\n{edges}\n"
        context += "\n"

    # Construct the final prompt
    prompt = f"{system_message}\n\n{context}\nQuestion: {question}\n\nAnswer:"

    return prompt

def generate_response(client, question):
  from retrieve import retrieval

  sub_graphs, descriptions = retreival(question, k=3)
  rag_prompt = construct_rag_prompt(question, descriptions)

  # Now you can use this prompt with your LLM
  completion = client.chat.completions.create(
      model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
      messages=[
          {"role": "system", "content": "You are a medical knowledge assistant."},
          {"role": "user", "content": rag_prompt}
      ],
      temperature=0.6,
      top_p=0.95,
      max_tokens=4096,
      frequency_penalty=0,
      presence_penalty=0,
      stream=True
  )

  #   words to delete from the response:
  to_delete = [
    r"\(Knowledge Graph \d+\)",
    r"Knowledge Graph \d+"
  ]

  # Print the streaming response and save the response
  full_response = ""
  for chunk in completion:
    if chunk.choices[0].delta.content is not None:
      content = chunk.choices[0].delta.content
      for word in to_delete:
          content = re.sub(word, "", content)
      # print(content, end="")
      full_response += content
  return full_response


# # Example usage:
# questions = [
#     "How does air pollution impact the treatment or worsening of asthma and COPD symptoms?",
#     "How does air pollution impact the treatment or worsening of COPD symptoms?",
#     "How does air pollution impact the treatment or worsening of asthma symptoms?",
#     "What does asthma mean?",
#     "What is the color of Zied's shoes?"
# ]

# for question in questions:
#     print(f"\n\nQuestion: {question}\n")
#     # Get the response
#     full_response = generate_response(question)
#     print(f"Response: {full_response}\n")