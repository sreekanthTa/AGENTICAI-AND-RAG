import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ------------------------
# 1️⃣ Query
# ------------------------
query = "Explain JavaScript loops"

# ------------------------
# 2️⃣ Embedding model
# ------------------------
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorize_query = embeddings_model.embed_query(query)

# ------------------------
# 3️⃣ Qdrant client
# ------------------------
client = QdrantClient(url="http://localhost:6333")
collection_name = "pdf_embeddings"
vector_size = 384  # embedding dimension

try:
    client.get_collection(collection_name)
except:
    print(f"Collection '{collection_name}' not found!")

# ------------------------
# 4️⃣ Retrieve top 3 chunks from Qdrant
# ------------------------
results = client.search(
    collection_name=collection_name,
    query_vector=vectorize_query,
    limit=3
)

print("\n--- Top results ---")
for i, result in enumerate(results):
    print(f"\nResult {i+1} (Score: {result.score:.4f}):\n{result.payload['text']}")

# ------------------------
# 5️⃣ LLM setup
# ------------------------
llm = ChatOpenAI(
    temperature=0,
    # openai_api_key=os.getenv("GROQ_API_KEY"),
    model_name="gpt-3.5-turbo"  # must be a chat model
)

# ------------------------
# 6️⃣ Prompt template
# ------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are an AI JavaScript Expert and Tutor.
Answer the user question using the following context:

{context}

User question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT_TEMPLATE
)

# ------------------------
# 7️⃣ Create LLMChain
# ------------------------
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# ------------------------
# 8️⃣ Combine retrieved text as context
# ------------------------
context = "\n".join([res.payload["text"] for res in results])

# ------------------------
# 9️⃣ Generate answer
# ------------------------
answer = chain.run(context=context, question=query)
print("\n--- Generated Answer ---")
print(answer)
