# Configuration file for Document Analysis System

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"  # Ultra-lightweight model under 1GB: gemma3:1b, phi3:mini
OLLAMA_TIMEOUT = 15  # Even faster timeout for 1B model

# RAG Pipeline Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for CPU (80MB)
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# Processing Configuration
CHUNK_SIZE = 800  # Reduced for faster processing
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_SECTIONS = 8  # Reduced for speed
MAX_SUBSECTIONS = 5
TOP_K_RETRIEVAL = 10  # Reduced for faster retrieval

# Performance Configuration
MAX_PROCESSING_TIME = 60  # seconds
MAX_MEMORY_USAGE = 1024  # MB
CPU_THREADS = 4
BATCH_SIZE = 5  # Process documents in smaller batches

# Document Processing
SUPPORTED_FORMATS = [".pdf"]
MAX_FILE_SIZE = 50  # MB
MAX_DOCUMENTS = 10

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TIMEOUT = 120

# Paths
DOCUMENTS_DIR = "./documents"
OUTPUT_DIR = "./output"
TEMP_DIR = "./temp"
