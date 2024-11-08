from loaders.dataloader import Loader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from loaders.dataloader import Loader
from chromadb.config import Settings

load_dotenv()

vectordb = Chroma(
    collection_name="pet-insurance",
    persist_directory="data/db",
    embedding_function=OpenAIEmbeddings(),
)

for i in range (len(vectordb.))