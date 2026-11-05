from pypdf import PdfReader
from langchain_text_splitters import(
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import os 
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder

# 0. Setup
load_dotenv()
client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)
MODEL = "openai/gpt-oss-20b"


# Read the pdf
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [t for t in pdf_texts if t] # Remove empty pages

# First split: bigger pieces[1000 characters]
character_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", ". ", " ", ""], chunk_size = 1000, chunk_overlap = 0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Second split : small chunks (256 tokens) that the embedding model can handle
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap = 0, tokens_per_chunk = 256
)
token_split_texts = []

for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print("Total_chunks: ", len(token_split_texts))

# -----Stores chunks in chromadb (Embeddings are created automatically)-----
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    "microsoft_collection", embedding_function = embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
collection.add(ids = ids , documents = token_split_texts)

# 3. Query Expansion (ask the llm for related questions)
def generate_multi_query(query):
    system_prompt = (
        "You are a knowledgeable financial research assistant. "
        "Your users are enquiring about an annual report. "
        "For the given question, propose up to five related questions. "
        "Make them concise, single-topic and complete. "
        "List each question on a separate line without numbering. "
    )
    response = client.chat.completions.create(
        model = MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )
    lines = response.choices[0].message.content.split("\n")
    return [line.strip() for line in lines if line.strip()] # Drop blank lines

original_query = "What were the most important factors that contributed to increases in revenue?"
generated_queries = generate_multi_query(original_query)
joint_queries = [original_query] + generated_queries

print("\nQueries Used:")
for q in joint_queries:
    print("-", q)

# Retrieve: Search with all queries at once
results = collection.query(query_texts = joint_queries, n_results = 10)

# Remove duplicates
unique_documents = list({doc for docs in results["documents"] for doc in docs})
print("\nUnique chunks retrieved: ", len(unique_documents))

# 5. Re rank : score every (query, chunk) pair with cross encoder.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# IMPORTANT: always score against the original query.
pairs = [[original_query, doc] for doc in unique_documents]
scores = cross_encoder.predict(pairs)

# argsort gives indexes from low to high score; [::-1] reverses it (best first)
ranked_indexes = np.argsort(scores)[::-1]

print("====RERANKED TOP 5 ====")
for i in ranked_indexes[:5]:
    print(f"Score : {scores[i]:.2f} | {unique_documents[i][:150]}...\n")

# 6. generate the final answer using the top 5 chunks
top_documents = [unique_documents[i] for i in ranked_indexes[:5]]
context = "\n\n".join(top_documents)

response = client.chat.completions.create(
    model = MODEL,
    messages=[
        {"role": "system", 
         "content": "You are a financial research assistant. Answer only from the given context."
         },
        {"role": "user",
         "content": f"Context:\n\n{context}\n\nQuestion: {original_query}"
         },
    ]
)
print("===Final Answer ====")
print(response.choices[0].message.content)