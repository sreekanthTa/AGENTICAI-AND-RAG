from  uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

file_path = "./javascript_tutorial.pdf"
loader = PyPDFLoader(file_path)

# LOAD DOCUMENTS 
documents = loader.load()

documents[0]
print(documents[10])

# CHUNK SPLITTER FOR ABOVE LOADED DOCUMENTS
chunk_size = 500
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
)

chunks = text_splitter.split_documents(documents)

# EMBEDDING MODEL TO CREATE EMBEDDINGS FOR ABOVE CREATED CHUNKS
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# QDRANT VECTOR DB :
client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
collection_name = "pdf_embeddings"
vector_size = 384  # MiniLM-L6-v2 embedding dimension

try:
    client.get_collection(collection_name)
except:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

# INSERT INTO QUDRANT VECTOR DB
points = []

for chunk in chunks:
    vector_embeding_for_a_chunk  = embeddings_model.embed_query(chunk.page_content)
    points.append({
       "id":str(uuid4()),
       "vector":vector_embeding_for_a_chunk,
       "payload": {"text": chunk.page_content}
    })

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"✅ PDF loaded and {len(points)} chunks inserted into Qdrant collection '{collection_name}'")
