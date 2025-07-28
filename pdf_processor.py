import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class DocumentSection:
    title: str
    content: str
    page_number: int
    start_char: int
    end_char: int
    level: int  # heading level (1=h1, 2=h2, etc.)

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF with page information"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'char_count': len(text)
                })
            
            doc.close()
            return {
                'pages': pages_text,
                'total_pages': len(pages_text),
                'total_chars': sum(p['char_count'] for p in pages_text)
            }
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return {'pages': [], 'total_pages': 0, 'total_chars': 0}
    
    def identify_sections(self, pages_text: List[Dict]) -> List[DocumentSection]:
        """Identify sections and subsections in the document with content-aware patterns"""
        sections = []
        
        # Dynamic patterns based on content type
        for page_info in pages_text:
            page_num = page_info['page_number']
            text = page_info['text']
            
            # Determine document type for better pattern matching
            doc_type = self._identify_document_type(text)
            section_patterns = self._get_section_patterns(doc_type)
            
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Check if line matches section patterns
                is_section = False
                level = 1
                
                for level_num, pattern in enumerate(section_patterns, 1):
                    if re.match(pattern, line, re.IGNORECASE):
                        is_section = True
                        level = level_num
                        break
                
                # Additional heuristics for section detection
                if not is_section:
                    is_section = self._is_likely_section_title(line, lines, i)
                
                if is_section:
                    # Extract content for this section
                    content = self._extract_section_content(lines, i, section_patterns)
                    
                    # Create better titles based on content
                    improved_title = self._improve_section_title(line, content, doc_type)
                    
                    sections.append(DocumentSection(
                        title=improved_title,
                        content=content,
                        page_number=page_num,
                        start_char=0,
                        end_char=len(content),
                        level=level
                    ))
        
        return sections
    
    def _identify_document_type(self, text: str) -> str:
        """Identify the type of document to apply appropriate section patterns"""
        text_lower = text.lower()
        
        # Food/Recipe documents
        food_indicators = ['recipe', 'ingredient', 'cook', 'bake', 'preparation', 'serve', 'dish', 
                          'meal', 'breakfast', 'lunch', 'dinner', 'appetizer', 'dessert', 'side']
        
        # Technical/Manual documents
        tech_indicators = ['configure', 'setup', 'install', 'procedure', 'step', 'method', 
                          'process', 'system', 'software', 'application']
        
        # Academic/Research documents
        academic_indicators = ['abstract', 'methodology', 'results', 'conclusion', 'research', 
                              'study', 'analysis', 'findings']
        
        # Business/Professional documents
        business_indicators = ['strategy', 'plan', 'objective', 'goal', 'management', 'policy', 
                              'procedure', 'guideline']
        
        if any(indicator in text_lower for indicator in food_indicators):
            return 'food'
        elif any(indicator in text_lower for indicator in tech_indicators):
            return 'technical'
        elif any(indicator in text_lower for indicator in academic_indicators):
            return 'academic'
        elif any(indicator in text_lower for indicator in business_indicators):
            return 'business'
        else:
            return 'general'
    
    def _get_section_patterns(self, doc_type: str) -> List[str]:
        """Get section patterns based on document type"""
        base_patterns = [
            r'^[A-Z][A-Z\s]*$',  # ALL CAPS
            r'^\d+\.?\s+[A-Z].*',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
        ]
        
        if doc_type == 'food':
            return [
                r'^(?:Ingredients?|Preparation|Instructions?|Directions?|Method|Recipe|Serving|Garnish).*$',
                r'^(?:Breakfast|Lunch|Dinner|Appetizer|Main|Side|Dessert|Drink).*$',
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Recipe)?$',
            ] + base_patterns
        elif doc_type == 'technical':
            return [
                r'^(?:Overview|Setup|Configuration|Installation|Procedure|Steps?).*$',
                r'^(?:Requirements?|Prerequisites?|Getting Started).*$',
                r'^(?:Step\s+\d+|Phase\s+\d+).*$',
            ] + base_patterns
        elif doc_type == 'academic':
            return [
                r'^(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References?).*$',
                r'^(?:Background|Literature Review|Methodology|Findings).*$',
            ] + base_patterns
        else:
            return base_patterns
    
    def _is_likely_section_title(self, line: str, lines: List[str], index: int) -> bool:
        """Use heuristics to determine if a line is likely a section title"""
        # Title-like characteristics
        if len(line) > 100:  # Too long to be a title
            return False
        
        if len(line) < 5:  # Too short to be meaningful
            return False
        
        # Check if it's followed by content
        has_following_content = False
        for i in range(index + 1, min(index + 5, len(lines))):
            if lines[i].strip() and len(lines[i].strip()) > 20:
                has_following_content = True
                break
        
        # Check capitalization patterns
        words = line.split()
        if len(words) > 1:
            capitalized_words = sum(1 for word in words if word[0].isupper())
            if capitalized_words >= len(words) / 2:  # At least half the words are capitalized
                return has_following_content
        
        return False
    
    def _extract_section_content(self, lines: List[str], start_index: int, patterns: List[str]) -> str:
        """Extract content for a section"""
        content_lines = []
        
        for i in range(start_index + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Stop if we hit another section
            is_next_section = any(re.match(p, line, re.IGNORECASE) for p in patterns)
            if is_next_section or self._is_likely_section_title(line, lines, i):
                break
            
            content_lines.append(line)
            if len(' '.join(content_lines)) > 600:  # Limit content length
                break
        
        return ' '.join(content_lines)
    
    def _improve_section_title(self, original_title: str, content: str, doc_type: str) -> str:
        """Improve section title based on content and document type"""
        if not content or len(content) < 20:
            return original_title
        
        content_lower = content.lower()
        
        # Generic fallback title improvement
        if original_title.lower() in ['working with documents', 'section', 'content']:
            # Extract meaningful title from content
            if doc_type == 'food':
                return self._extract_food_title(content)
            else:
                return self._extract_generic_title(content)
        
        return original_title
    
    def _extract_food_title(self, content: str) -> str:
        """Extract food-related title from content"""
        content_lower = content.lower()
        
        # Look for dish names and food terms
        food_terms = []
        
        # Common food words that indicate dish types
        if any(term in content_lower for term in ['salad', 'lettuce', 'greens']):
            food_terms.append('Salad')
        if any(term in content_lower for term in ['soup', 'broth', 'simmer']):
            food_terms.append('Soup')
        if any(term in content_lower for term in ['pasta', 'spaghetti', 'noodles']):
            food_terms.append('Pasta')
        if any(term in content_lower for term in ['rice', 'grain', 'pilaf']):
            food_terms.append('Rice Dish')
        if any(term in content_lower for term in ['wrap', 'tortilla', 'roll']):
            food_terms.append('Wraps')
        if any(term in content_lower for term in ['dumpling', 'wrapper', 'fold']):
            food_terms.append('Dumplings')
        if any(term in content_lower for term in ['vegetarian', 'vegan', 'plant-based']):
            food_terms.append('Vegetarian')
        if any(term in content_lower for term in ['gluten-free', 'gluten free']):
            food_terms.append('Gluten-Free')
        
        if food_terms:
            return ' '.join(food_terms[:2])  # Combine up to 2 terms
        
        # Extract first meaningful food-related phrase
        sentences = content.split('.')
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 60:
                return sentence
        
        return 'Recipe Instructions'
    
    def _extract_generic_title(self, content: str) -> str:
        """Extract generic meaningful title from content"""
        # Get first substantial sentence
        sentences = content.split('.')
        for sentence in sentences[:2]:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 60:
                return sentence
        
        # Fallback: extract key words
        words = content.split()[:10]
        meaningful_words = [word for word in words if len(word) > 3 and word.isalpha()]
        if meaningful_words:
            return ' '.join(meaningful_words[:4]).title()
        
        return 'Document Section'
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for RAG processing (optimized for speed)"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Simple break at sentence boundaries (faster than complex logic)
            if end < len(text) and '.' in chunk:
                last_period = chunk.rfind('.')
                if last_period > start + chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
