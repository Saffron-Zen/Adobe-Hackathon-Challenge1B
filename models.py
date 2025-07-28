from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class ChallengeInfo(BaseModel):
    challenge_id: str
    test_case_name: str

class Document(BaseModel):
    filename: str
    title: str

class Persona(BaseModel):
    role: str

class JobToBeDone(BaseModel):
    task: str

class InputData(BaseModel):
    challenge_info: ChallengeInfo
    documents: List[Document]
    persona: Persona
    job_to_be_done: JobToBeDone

class ExtractedSection(BaseModel):
    document: str
    section_title: str
    importance_rank: int
    page_number: int

class SubsectionAnalysis(BaseModel):
    document: str
    refined_text: str
    page_number: int

class Metadata(BaseModel):
    input_documents: List[str]
    persona: str
    job_to_be_done: str
    processing_timestamp: str

class OutputData(BaseModel):
    metadata: Metadata
    extracted_sections: List[ExtractedSection]
    subsection_analysis: List[SubsectionAnalysis]
