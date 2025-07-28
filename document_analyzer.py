import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Any
import logging

from pdf_processor import PDFProcessor, DocumentSection
from rag_pipeline import RAGPipeline
from ollama_client import OllamaClient
from models import InputData, OutputData, Metadata, ExtractedSection, SubsectionAnalysis
from optimized_config import determine_processing_mode, get_mode_config
import config

class DocumentAnalyzer:
    def __init__(self, ollama_model: str = None, documents_dir: str = None, 
                 processing_mode: str = "balanced", enable_optimizations: bool = True):
        """Initialize the document analyzer with performance optimizations"""
        self.logger = logging.getLogger(__name__)
        self.documents_dir = documents_dir or config.DOCUMENTS_DIR
        self.processing_mode = processing_mode
        self.enable_optimizations = enable_optimizations
        
        # Get mode configuration
        self.mode_config = get_mode_config(processing_mode)
        
        # Use config values or defaults
        model = ollama_model or config.OLLAMA_MODEL
        
        # Initialize components with optimizations
        try:
            self.pdf_processor = PDFProcessor()
            
            # Initialize RAG pipeline with optimizations
            self.rag_pipeline = RAGPipeline(
                model_name=config.EMBEDDING_MODEL,
                persist_directory=config.VECTOR_DB_PATH,
                batch_size=self.mode_config.get('batch_size', 32),
                enable_optimizations=enable_optimizations
            )
            
            # Initialize Ollama client with optimizations
            self.ollama_client = OllamaClient(
                base_url=config.OLLAMA_BASE_URL,
                model=model,
                max_workers=self.mode_config.get('max_workers', 4),
                enable_cache=enable_optimizations,
                processing_mode=processing_mode
            )
            
            self.logger.info(f"Initialized all components successfully with model: {model}, mode: {processing_mode}")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def process_documents(self, input_data: InputData) -> OutputData:
        """Main processing function"""
        start_time = time.time()
        
        try:
            # Extract text from all PDFs
            self.logger.info("Starting document processing...")
            all_documents = []
            
            for doc_info in input_data.documents:
                doc_path = os.path.join(self.documents_dir, doc_info.filename)
                if not os.path.exists(doc_path):
                    self.logger.warning(f"Document not found: {doc_path}")
                    continue
                
                # Extract text and sections
                pdf_data = self.pdf_processor.extract_text_from_pdf(doc_path)
                sections = self.pdf_processor.identify_sections(pdf_data['pages'])
                
                # Prepare documents for RAG
                for section in sections:
                    chunks = self.pdf_processor.chunk_text(section.content)
                    for chunk in chunks:
                        all_documents.append({
                            'document': doc_info.filename,
                            'content': chunk,
                            'page_number': section.page_number,
                            'section_title': section.title,
                            'title': doc_info.title
                        })
            
            self.logger.info(f"Processed {len(all_documents)} document chunks")
            
            # Add documents to RAG pipeline
            self.rag_pipeline.add_documents(all_documents)
            
            # Get relevant sections using RAG (reduced for speed)
            relevant_docs = self.rag_pipeline.get_context_for_persona_job(
                input_data.persona.role,
                input_data.job_to_be_done.task,
                top_k=10  # Reduced from 15 for faster processing
            )
            
            # Analyze and rank sections using Ollama (parallel processing for speed)
            ranked_sections = self.ollama_client.rank_sections(
                relevant_docs[:8],  # Limit to top 8 for speed
                input_data.persona.role,
                input_data.job_to_be_done.task
            )
            
            # Generate output
            output_data = self._create_output(
                input_data,
                ranked_sections,
                start_time
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            # Optimize memory usage after processing
            if self.enable_optimizations:
                self.ollama_client.optimize_memory_usage()
                self.rag_pipeline.optimize_memory()
                self.logger.info("Memory optimization completed")
            
            return output_data
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    def _create_output(self, input_data: InputData, ranked_sections: List[Dict], 
                      start_time: float) -> OutputData:
        """Create the output data structure"""
        
        # Create metadata
        metadata = Metadata(
            input_documents=[doc.filename for doc in input_data.documents],
            persona=input_data.persona.role,
            job_to_be_done=input_data.job_to_be_done.task,
            processing_timestamp=datetime.now().isoformat()
        )
        
        # Create extracted sections (top 10)
        extracted_sections = []
        for section in ranked_sections[:10]:
            # Use contextual title from analysis if available, with generic improvement
            contextual_title = section.get('analysis', {}).get('contextual_title')
            original_title = section.get('section_title', 'Untitled')
            
            # Apply generic title improvement
            section_title = self._improve_section_title(
                contextual_title or original_title, 
                section.get('content', ''),
                original_title
            )
            
            extracted_sections.append(ExtractedSection(
                document=section['document'],
                section_title=section_title,
                importance_rank=section['importance_rank'],
                page_number=section['page_number']
            ))
        
        # Create subsection analysis (top 5 with refined text)
        subsection_analysis = []
        for section in ranked_sections[:5]:
            try:
                refined_text = self.ollama_client.extract_refined_text(
                    section['content'],
                    input_data.persona.role,
                    input_data.job_to_be_done.task
                )
                
                # Ensure we have some content
                if not refined_text or len(refined_text.strip()) < 10:
                    # Use truncated original content as fallback
                    refined_text = section['content'][:200] + "..." if len(section['content']) > 200 else section['content']
                
                subsection_analysis.append(SubsectionAnalysis(
                    document=section['document'],
                    refined_text=refined_text,
                    page_number=section['page_number']
                ))
                
                self.logger.info(f"Created subsection analysis for {section['document']} page {section['page_number']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create subsection analysis: {str(e)}")
                # Add with original content as fallback
                fallback_text = section['content'][:200] + "..." if len(section['content']) > 200 else section['content']
                subsection_analysis.append(SubsectionAnalysis(
                    document=section['document'],
                    refined_text=fallback_text or "Content analysis not available.",
                    page_number=section['page_number']
                ))
        
        return OutputData(
            metadata=metadata,
            extracted_sections=extracted_sections,
            subsection_analysis=subsection_analysis
        )
    
    def _improve_section_title(self, title: str, content: str, original_title: str) -> str:
        """Generic method to improve section titles based on content - now more aggressive"""
        if not title or len(title.strip()) < 3:
            title = original_title or "Document Section"
        
        # If we have content, always try to generate a better title
        if content and len(content.strip()) > 30:
            # First, check if title is generic or unhelpful
            if self._is_generic_or_poor_title(title):
                improved_title = self._extract_complete_title_from_content(content, title)
                if improved_title and len(improved_title) > 5:
                    return improved_title
            
            # Even if title isn't generic, try to improve it based on content
            enhanced_title = self._enhance_title_with_content_context(title, content)
            if enhanced_title and enhanced_title != title:
                return enhanced_title
        
        # Clean and format the existing title
        return self._clean_and_format_title(title)
    
    def _is_generic_or_poor_title(self, title: str) -> bool:
        """Check if a title is generic, poor, or unhelpful"""
        title_lower = title.lower().strip()
        
        # Obviously generic titles
        generic_titles = [
            'working with documents', 'section', 'untitled', 'content', 'text', 
            'part', 'chapter', 'document section', 'page', 'introduction'
        ]
        
        if any(generic in title_lower for generic in generic_titles):
            return True
        
        # Titles that are just fragments or incomplete
        if len(title.strip()) < 10:
            return True
        
        # Titles that end abruptly (incomplete extraction)
        if title.endswith('...') or title.endswith('..'):
            return True
        
        # Check for meaningless patterns
        if re.match(r'^[o]\s+', title_lower):  # Starts with "O " (likely OCR error)
            return True
        
        return False
    
    def _enhance_title_with_content_context(self, title: str, content: str) -> str:
        """Enhance existing title with context from content"""
        import re
        
        # Detect content type from the content itself
        content_lower = content.lower()
        
        # Food content indicators
        food_indicators = [
            'ingredient', 'recipe', 'cook', 'bake', 'serve', 'dish', 'meal',
            'onion', 'garlic', 'salt', 'pepper', 'oil', 'sauce', 'flour',
            'vegetarian', 'gluten-free', 'buffet', 'spoon', 'mix', 'heat',
            'minutes', 'temperature', 'oven', 'pan', 'stir', 'simmer'
        ]
        
        # If it's food content, create food-specific titles
        if any(indicator in content_lower for indicator in food_indicators):
            return self._create_food_specific_title(content, title)
        
        # For other content types, extract meaningful phrases
        return self._create_contextual_title(content, title)
    
    def _create_food_specific_title(self, content: str, fallback_title: str) -> str:
        """Create food-specific titles based on content"""
        content_lower = content.lower()
        
        # Map content to food categories
        food_categories = {
            'wraps': ['wrapper', 'roll', 'wrap', 'tortilla', 'spring roll'],
            'salad': ['salad', 'lettuce', 'cucumber', 'tomato', 'greens', 'vinegar'],
            'soup': ['soup', 'broth', 'simmer', 'liquid', 'stock'],
            'pasta': ['pasta', 'noodle', 'spaghetti', 'linguine'],
            'rice_dish': ['rice', 'grain', 'pilaf', 'risotto'],
            'dumplings': ['dumpling', 'filling', 'wrapper', 'fold'],
            'stir_fry': ['stir', 'fry', 'wok', 'sauté'],
            'vegetarian': ['vegetarian', 'vegan', 'plant-based'],
            'gluten_free': ['gluten-free', 'gluten free'],
            'appetizer': ['appetizer', 'starter', 'small plate'],
            'side_dish': ['side', 'accompaniment', 'complement']
        }
        
        # Find matching categories
        matched_categories = []
        for category, indicators in food_categories.items():
            if any(indicator in content_lower for indicator in indicators):
                matched_categories.append(category)
        
        # Create title based on categories
        if matched_categories:
            title_parts = []
            
            # Add dietary restrictions first
            if 'vegetarian' in matched_categories:
                title_parts.append('Vegetarian')
            if 'gluten_free' in matched_categories:
                title_parts.append('Gluten-Free')
            
            # Add dish type
            dish_types = ['wraps', 'salad', 'soup', 'pasta', 'rice_dish', 'dumplings', 'stir_fry']
            for dish_type in dish_types:
                if dish_type in matched_categories:
                    formatted_name = dish_type.replace('_', ' ').title()
                    title_parts.append(formatted_name)
                    break
            
            # Add course type if relevant
            if 'appetizer' in matched_categories:
                title_parts.append('Appetizer')
            elif 'side_dish' in matched_categories:
                title_parts.append('Side Dish')
            
            if title_parts:
                return ' '.join(title_parts)
        
        # Fallback: extract recipe name from content
        return self._extract_recipe_name_from_content(content) or fallback_title
    
    def _extract_recipe_name_from_content(self, content: str) -> str:
        """Extract recipe name from content"""
        # Look for food-related phrases in the first few sentences
        sentences = content.split('.')[:3]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 60:
                # Check if it contains food words
                if any(word in sentence.lower() for word in ['cook', 'prepare', 'mix', 'serve', 'add', 'heat']):
                    # Clean and return as recipe instruction
                    cleaned = re.sub(r'^[o]\s+', '', sentence, flags=re.IGNORECASE)  # Remove OCR errors
                    if len(cleaned) > 10:
                        return cleaned.strip()
        
        return None
    
    def _create_contextual_title(self, content: str, fallback_title: str) -> str:
        """Create contextual title for non-food content"""
        # Extract the most informative sentence or phrase
        sentences = content.split('.')
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 70:
                # Clean up common OCR errors
                sentence = re.sub(r'^[o]\s+', '', sentence, flags=re.IGNORECASE)
                if len(sentence) > 10:
                    return sentence
        
        # Fallback to extracting key phrases
        words = re.findall(r'\b[A-Za-z]{4,}\b', content[:200])
        if len(words) >= 3:
            return ' '.join(words[:4]).title()
        
        return fallback_title
    
    def _extract_complete_title_from_content(self, content: str, current_title: str) -> str:
        """Extract a complete, meaningful title from content that relates to the document"""
        import re
        
        # Clean content
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Look for Acrobat/PDF-specific instructional content
        acrobat_patterns = [
            # Acrobat tool and feature usage
            r'((?:Use|Using|Access|Open)\s+(?:the\s+)?(?:Fill & Sign|Prepare Forms|Digital Signatures?|Export|Share|Edit)[^.]{0,50}(?:tool|feature|option|menu)?)',
            # Form-related instructions
            r'((?:Create|Creating|Build|Building|Design|Designing)\s+[^.]{5,60}(?:forms?|documents?|PDFs?))',
            # Specific Acrobat procedures
            r'((?:How to|Learn to|Steps to)\s+[^.]{10,70}(?:in Acrobat|with PDF|using forms))',
            # Feature descriptions with context
            r'([A-Z][^.]{15,70}(?:allows you to|enables you to|helps you)\s+[^.]{10,50})',
        ]
        
        for pattern in acrobat_patterns:
            matches = re.findall(pattern, content[:600], re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                candidate = self._clean_document_title(candidate)
                if 15 <= len(candidate) <= 80 and self._is_relevant_title(candidate):
                    return candidate
        
        # Look for procedural content that explains processes
        procedure_patterns = [
            r'([A-Z][^.]{20,70}(?:process|procedure|workflow|method|technique))',
            r'([A-Z][^.]{15,60}(?:allows|enables|provides|offers)\s+[^.]{10,40})',
            r'((?:Interactive|Fillable|Digital|Electronic)\s+[^.]{10,60})',
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, content[:500], re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                candidate = self._clean_document_title(candidate)
                if 20 <= len(candidate) <= 80 and self._is_relevant_title(candidate):
                    return candidate
        
        # Look for educational content that teaches something
        teaching_patterns = [
            r'([A-Z][^.]{25,80}(?:forms?|documents?|signatures?|fields?|PDFs?))',
            r'([A-Z][^.]{20,70}(?:data|information|content|text))',
        ]
        
        for pattern in teaching_patterns:
            matches = re.findall(pattern, content[:400], re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                candidate = self._clean_document_title(candidate)
                if 25 <= len(candidate) <= 80 and self._is_relevant_title(candidate):
                    return candidate
        
        return None
    
    def _clean_document_title(self, title: str) -> str:
        """Clean a title to make it more document-appropriate"""
        import re
        
        # Remove common prefixes that don't add value
        prefixes_to_remove = [
            r'^(?:The|A|An)\s+', r'^(?:This|That)\s+', r'^(?:Here|There)\s+',
            r'^(?:You can|Users can|One can)\s+'
        ]
        
        for prefix in prefixes_to_remove:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Remove incomplete endings
        suffixes_to_remove = [
            r'\s+(?:the|a|an|and|or|of|in|on|at|to|for|with|by)$',
            r'\s+(?:that|which|who|when|where|how)$'
        ]
        
        for suffix in suffixes_to_remove:
            title = re.sub(suffix, '', title, flags=re.IGNORECASE)
        
        # Clean up spacing and formatting
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Ensure proper capitalization
        if title:
            title = title[0].upper() + title[1:]
        
        return title
    
    def _is_relevant_title(self, title: str) -> bool:
        """Check if a title is relevant to document/PDF/Acrobat content"""
        title_lower = title.lower()
        
        # Relevant keywords for PDF/Acrobat documentation
        relevant_keywords = [
            'acrobat', 'pdf', 'form', 'document', 'sign', 'signature', 'fill',
            'create', 'edit', 'export', 'share', 'field', 'interactive', 'digital',
            'convert', 'tool', 'feature', 'prepare', 'validate', 'review'
        ]
        
        # Must contain at least one relevant keyword
        has_relevant = any(keyword in title_lower for keyword in relevant_keywords)
        
        # Exclude overly generic or unhelpful titles
        exclude_keywords = [
            'this section', 'the following', 'as shown', 'click here',
            'see above', 'see below', 'refer to', 'shown in'
        ]
        
        has_excluded = any(exclude in title_lower for exclude in exclude_keywords)
        
        return has_relevant and not has_excluded
    
    def _clean_and_format_title(self, title: str) -> str:
        """Clean and format section title properly"""
        if not title:
            return "Document Section"
        
        import re
        
        # Remove excessive whitespace and special characters
        title = re.sub(r'\s+', ' ', title.strip())
        title = title.replace('\n', ' ').replace('\r', '')
        
        # Remove quotes if they wrap the entire title
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        
        # Fix incomplete titles (ending with incomplete words)
        incomplete_endings = [' the', ' a', ' an', ' and', ' or', ' of', ' in', ' on', ' at', ' to']
        for ending in incomplete_endings:
            if title.lower().endswith(ending):
                title = title[:-len(ending)].strip()
        
        # Fix incomplete beginnings
        incomplete_beginnings = ['the ', 'a ', 'an ']
        for beginning in incomplete_beginnings:
            if title.lower().startswith(beginning):
                # Only remove if what remains is substantial
                remaining = title[len(beginning):].strip()
                if len(remaining) > 5:
                    title = remaining
        
        # Ensure title length is appropriate
        if len(title) > 60:
            # Find a good breaking point
            words = title.split()
            if len(words) > 8:
                title = ' '.join(words[:8]) + "..."
            else:
                title = title[:57] + "..."
        
        # Ensure minimum length
        if len(title) < 5:
            title = "Document Section"
        
        # Proper capitalization - capitalize each word appropriately
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def process_from_json(self, input_json_path: str, output_json_path: str):
        """Process documents from JSON input and save to JSON output"""
        try:
            # Load input
            with open(input_json_path, 'r') as f:
                input_dict = json.load(f)
            
            input_data = InputData(**input_dict)
            
            # Process
            output_data = self.process_documents(input_data)
            
            # Save output
            with open(output_json_path, 'w') as f:
                json.dump(output_data.dict(), f, indent=2)
            
            self.logger.info(f"Results saved to {output_json_path}")
            
        except Exception as e:
            self.logger.error(f"JSON processing failed: {str(e)}")
            raise

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == "__main__":
    setup_logging()
    
    # Create analyzer
    analyzer = DocumentAnalyzer()
    
    # Process from command line arguments or default files
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    
    try:
        analyzer.process_from_json(input_file, output_file)
        print(f"Processing completed successfully. Output saved to {output_file}")
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        sys.exit(1)
