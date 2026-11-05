import chromadb
import pandas as pd
from pypdf import PdfReader

def project_embeddings(embeddings, umap_transform):
    # Reduce embeddings to 2D using UMAP(used for plotting)
    return umap_transform.transform(embeddings)

def word_wrap(text, width = 87):
    #Split text into chunks of 'width' characters and join them with new lines
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def extract_text_from_pdf(file_path):
    # Read every page of the pdf and combine the text into one string
    text = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def load_chroma(filename, collection_name, embedding_function):
    # 1. Extract text from the pdf
    text = extract_text_from_pdf(filename)

    # 2. Split the text into paragraphs
    paragraphs = text.split("\n\n")

    # 3. Create an embedding for every paragraph
    embeddings = [embedding_function(p) for p in paragraphs]

    # 4. Build a table(DataFrame) with the text and its embeddings
    df = pd.DataFrame({"text": paragraphs, "embeddings": embeddings})

    # 5. Create a chroma collection and add the row one by one
    collection = chromadb.Client().create_collection(collection_name)
    for i, row in df.iterrows():
        collection.add(
            ids = str(i),
            documents = row["text"],
            embeddings = row["embeddings"],
        )
    return collection