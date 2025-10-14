import os
import pinecone
import logging
from google.cloud import aiplatform
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict

logging.basicConfig(level=logging.INFO)

# Initialize Pinecone and Vertex AI clients
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west1-gcp')
index = pinecone.Index("rag-kb-index")
llm_client = aiplatform.gapic.PredictionServiceClient()

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int
    question: str
    snippets: list
    critique: str
    final_answer: str

def embed_text(text):
    response = llm_client.embed_text(
        model="models/gemini-embedding-001",
        instances=[{"content": text}]
    )
    return response.predictions[0]["embedding"]

def retrieve_kb(question):
    q_emb = embed_text(question)
    res = index.query(vector=q_emb, top_k=5, include_metadata=True)
    snippets = [match["metadata"]["text"] for match in res["matches"]]
    logging.info(f"Retrieved snippets: {snippets}")
    return snippets

def generate_answer(question, snippets):
    prompt = (
        f"Question: {question}\n\n"
        f"Context snippets:\n{chr(10).join(snippets)}\n\n"
        "Please answer the question citing the snippets as [KBxxx]."
    )
    response = llm_client.predict(
        model_name="./locations/us-central1/publishers/google/models/gemini-1.5",
        instances=[{"content": prompt}],
        parameters={"temperature": 0}
    )
    answer = response.predictions[0]["content"]
    logging.info(f"Initial answer: {answer}")
    return answer

def critique_answer(answer, snippets):
    prompt = (
        f"Review the following answer:\n{answer}\n\n"
        f"Compare it against these snippets:\n{chr(10).join(snippets)}\n"
        "Respond with either 'COMPLETE' or 'REFINE:<missing keywords>'."
    )
    response = llm_client.predict(
        model_name="./locations/us-central1/publishers/google/models/gemini-1.5",
        instances=[{"content": prompt}],
        parameters={"temperature": 0}
    )
    critique = response.predictions[0]["content"].strip()
    logging.info(f"Critique: {critique}")
    return critique

def refine_answer(question, snippets, missing_keywords):
    missing_emb = embed_text(missing_keywords)
    extra_res = index.query(vector=missing_emb, top_k=1, include_metadata=True)
    extra_snippet = extra_res["matches"][0]["metadata"]["text"]
    new_snippets = snippets + [extra_snippet]
    logging.info(f"Refinement snippet added: {extra_snippet}")

    return generate_answer(question, new_snippets)


if __name__ == "__main__":
    question = "What are best practices for caching?"
    snippets = retrieve_kb(question)
    initial_answer = generate_answer(question, snippets)
    critique = critique_answer(initial_answer, snippets)

    if critique.startswith("REFINE:"):
        missing_kw = critique.split("REFINE:")[1].strip()
        final_answer = refine_answer(question, snippets, missing_kw)
    else:
        final_answer = initial_answer

    print(f"Final Answer:\n{final_answer}")
