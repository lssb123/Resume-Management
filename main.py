from fastapi import FastAPI, File, UploadFile, HTTPException
import pdfplumber
import os
import uuid
import tensorflow_hub as hub
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

# Qdrant setup
client = QdrantClient(url="http://localhost:6333")
collection_name = "resume_collection"

# TensorFlow Hub model for text embeddings
embed_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(embed_model_url)

# Check if the collection exists and create it if it does not
def ensure_collection_exists():
    try:
        # List existing collections
        collections = client.get_collections().collections
        if collection_name not in [col.name for col in collections]:
            # If collection does not exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)  # Adjust vector size as needed
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred while ensuring collection exists: {str(e)}")
        raise e


ensure_collection_exists()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def save_uploaded_file(uploaded_file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.file.read())
    return file_path

def extract_text_by_page(pdf_path):
    page_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_texts.append(page.extract_text())
    return page_texts

def generate_embeddings(texts):
    return embed_model(texts).numpy().tolist()

def store_in_qdrant(resume_name: str, page_content: str, page_number: int):
    embeddings = generate_embeddings([page_content])
    point_id = str(uuid.uuid4())
    point = PointStruct(
        id=point_id,
        vector=embeddings[0],
        payload={
            "resume_name": resume_name,
            "page_number": page_number,
            "content": page_content
        }
    )
    
    client.upsert(collection_name=collection_name, points=[point], wait=True)

def search_qdrant(technologies, top_k=5):
    try:
        # Generate embeddings for the input technologies (assuming it's a list of technologies)
        question_embedding = embed_model(technologies)  # Produces a 2D array
        technology_embeddings = np.array(question_embedding).tolist()[0]  # Get the first embedding (vector of size 512)

        # Ensure that the embedding is of correct size (512)
        if len(technology_embeddings) != 512:
            raise ValueError(f"Embedding size mismatch. Expected 512, got {len(technology_embeddings)}")

        # Perform search in QdrantDB
        search_results = client.search(
            collection_name=collection_name,
            query_vector=technology_embeddings,  # Vector with correct size
            limit=top_k  # Specify the number of top results (top_k)
        )
        # print(search_results)
        # Return search results
        return search_results

    except UnexpectedResponse as e:
        print(f"Unexpected response from Qdrant: {str(e)}")
        raise e
    except Exception as e:
        print(f"Error during Qdrant search: {str(e)}")
        raise e




# Function to extract unique person names from search results
def extract_unique_names(search_results):
    names = set()
    for result in search_results:
        # Assuming the 'payload' contains 'person_name'
        names.add(result.payload.get("resume_name"))
    return list(names)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_file_path = save_uploaded_file(file)
    page_texts = extract_text_by_page(pdf_file_path)
    resume_name = file.filename

    for page_number, page_content in enumerate(page_texts, start=1):
        if page_content:
            store_in_qdrant(resume_name, page_content, page_number)

    return {"status": "success", "message": "PDF content stored in Qdrant"}

# Define the request model
class TechnologyRequest(BaseModel):
    technologies: list[str]

@app.post("/search/")
async def search_technologies(request: TechnologyRequest):
    try:
        # print(request.technologies)
        search_results = search_qdrant(request.technologies)
        unique_names = extract_unique_names(search_results)
        return {"unique_names": unique_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
