🧠 Intelligent Document Analyzer
An intelligent system that analyzes a collection of documents, extracts the most relevant information based on a specific user persona and their "job-to-be-done," and presents it in a structured, prioritized format. This project is designed to run efficiently on local CPU environments without requiring internet access during execution.

✨ Key Features
Persona-Based Analysis: Tailors document analysis to the user's specific role and expertise.

Job-to-be-Done (JTBD) Focus: Prioritizes information that directly helps the user accomplish a specific task.

Efficient PDF Processing: Extracts text and identifies structural sections from multiple PDF documents.

Retrieval-Augmented Generation (RAG): Utilizes a RAG pipeline to find the most contextually relevant document chunks.

Local LLM Integration: Leverages a local Ollama instance with a lightweight model (gemma3:1b) for sophisticated analysis and text generation.

Fully Dockerized: Comes with a Dockerfile and docker-compose.yml for easy, consistent, and isolated deployment.

Optimized for CPU: Designed to run entirely on CPU with constraints on model size (≤ 1GB) and processing time.

Structured JSON Output: Generates a detailed JSON output with metadata, ranked sections, and refined text analysis.

🛠️ Technology Stack
Backend: Python

Containerization: Docker, Docker Compose

LLM Serving: Ollama

PDF Processing: PyMuPDF

Data Validation: Pydantic

Vector Embeddings & Search: Sentence-Transformers, ChromaDB

Caching: Custom Cache Manager

🚀 Getting Started
Prerequisites
Docker installed and running on your machine.

Ollama installed and running.

Pull the required lightweight model for Ollama:

ollama pull gemma3:1b

Installation & Execution
Clone the Repository

git clone https://github.com/your-username/intelligent-document-analyzer.git
cd intelligent-document-analyzer

Add Your Documents
Place the PDF files you want to analyze inside the documents/ directory.

Configure Your Analysis
Edit the input.json file to define the documents, persona, and job_to_be_done for your specific use case.

Build and Run with Docker Compose (Recommended)
This is the simplest way to get started. It builds the Docker image and runs the document analysis in one step.

docker-compose up --build

The analysis will start, and upon completion, the results will be saved to output.json in the project's root directory.

📊 Performance Analysis
Comparative Analysis: The optimized version consistently outperforms the baseline across all document scenarios in overall performance, speed, and resource efficiency.

Cross-lingual Latency: The optimized model shows significantly lower latency across multiple languages, with improvements ranging from 60% to over 85%.

Detailed Accuracy Analysis: The optimized model demonstrates superior precision, recall, and F1 scores, indicating a higher quality of extracted information.

Detailed Performance Analysis: The optimized system achieves faster processing times, lower memory usage, and higher throughput compared to the baseline.

Time Series Analysis: Over time, the optimized model maintains stable and superior performance in processing time, memory usage, and accuracy.

📝 Challenge Brief
This project is a solution for the following challenge: build a system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

Input Specification
Document Collection: 3-10 related PDFs. The solution must be generic enough to handle documents from any domain (e.g., research papers, financial reports, textbooks).

Persona Definition: A role description with specific expertise and focus areas (e.g., Researcher, Student, Investment Analyst).

Job-to-be-Done: A concrete task the persona needs to accomplish (e.g., "Prepare a literature review," "Analyze revenue trends").

Required Output
The output must be a JSON file (output.json) with the following structure:

Metadata:

input_documents: List of document filenames.

persona: The defined user persona.

job_to_be_done: The task to be accomplished.

processing_timestamp: The time the analysis was run.

Extracted Section: A list of objects, each containing:

document: The source document filename.

page_number: The page where the section was found.

section_title: The title of the extracted section.

importance_rank: The rank of the section's relevance.

Sub-section Analysis: A list of objects, each containing:

document: The source document filename.

refined_text: A concise, actionable summary of the subsection.

page_number: The page where the subsection was found.

Constraints
Must run on CPU only.

The language model size must be ≤ 1GB.

Processing time must be ≤ 60 seconds for a collection of 3-5 documents.

No internet access is allowed during execution.

Scoring Criteria
Criteria

Max Points

Description

Section Relevance

60

How well the selected sections match the persona and job requirements, with proper ranking.

Sub-Section Relevance

40

The quality of the granular subsection extraction and its refined summary.

📂 Project Structure
.
├── Dockerfile              # Defines the Docker image for the application
├── DOCKER_README.md        # Detailed instructions for Docker usage
├── cache_manager.py        # Caching logic to improve performance
├── config.py               # Main configuration for models, paths, and processing
├── docker-compose.yml      # Easy-to-use configuration for running with Docker Compose
├── document_analyzer.py    # The main script orchestrating the analysis pipeline
├── documents/              # Directory for input PDF files
│   └── ...
├── input.json              # Defines the analysis task (persona, job, documents)
├── models.py               # Pydantic models for data validation (input/output)
├── ollama_client.py        # Client to interact with the local Ollama LLM
├── optimized_config.py     # Performance-related configurations
└── README.md               # This file
