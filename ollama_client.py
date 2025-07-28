import json
import requests
import time
import asyncio
import aiohttp
import hashlib
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any, Optional, Generator
import logging
import re

from cache_manager import CacheManager
from collections import Counter
import re

# Basic English stopwords
_STOPWORDS = {
    'the','and','to','of','in','for','on','with','a','an','is','are','this','that',
    'it','as','at','by','from','or','be','was','were','will','has','have','not','but'
}
from optimized_config import OptimizedConfig, determine_processing_mode, get_mode_config

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:1b", 
                 max_workers: int = 4, enable_cache: bool = True, processing_mode: str = "balanced"):
        """Initialize Ollama client with performance optimizations"""
        self.base_url = base_url
        self.model = model
        self.max_workers = max_workers
        self.processing_mode = processing_mode
        self.config = get_mode_config(processing_mode)
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        self.cache_manager = CacheManager() if enable_cache else None
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session_lock = threading.Lock()
        
        # Test connection
        if not self._test_connection():
            raise ConnectionError("Cannot connect to Ollama server")
            
        self.logger.info(f"Initialized optimized Ollama client with model: {model}, mode: {processing_mode}")
        
        # Memory optimization settings
        self.embedding_cache = {}
        self.max_cache_size = OptimizedConfig.MAX_CACHE_SIZE
    
    def _test_connection(self) -> bool:
        """Test if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, context: str = "", 
                         max_tokens: int = -1, temperature: float = 0.1) -> str:
        """Generate response using Ollama (no token limit by default)"""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        # Build options dict
        options = {
            "temperature": temperature,
            "top_k": 40,  # Increased for better accuracy
            "top_p": 0.9,  # Higher for more diverse responses
            "repeat_penalty": 1.1,
            "stop": ["Human:", "Assistant:"]
        }
        
        # Only set num_predict if max_tokens is specified (> 0)
        if max_tokens > 0:
            options["num_predict"] = max_tokens
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": options
        }
        
        try:
            self.logger.debug(f"Sending request to Ollama with prompt length: {len(full_prompt)}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30  # Increased timeout for better results
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                self.logger.warning("Ollama returned empty response")
                return ""
            
            self.logger.debug(f"Ollama response length: {len(generated_text)}")
            return generated_text
            
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return ""
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to Ollama server")
            return ""
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return ""
    
    def generate_response_stream(self, prompt: str, callback=None) -> str:
        """Generate response with streaming for faster perceived performance"""
        try:
            full_prompt = prompt
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {"temperature": self.config.get('temperature', 0.1)}
                },
                stream=True,
                timeout=60
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            chunk = data['response']
                            full_response += chunk
                            if callback:
                                callback(chunk)  # Real-time processing
                    except json.JSONDecodeError:
                        continue
                        
            return full_response.strip()
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            return ""
    
    async def generate_response_async(self, prompt: str, context: str = "") -> str:
        """Async version of generate_response for parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.generate_response, 
            prompt, 
            context
        )
    
    async def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate multiple responses in parallel"""
        tasks = [self.generate_response_async(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def process_sections_parallel(self, sections: List[Dict]) -> List[Dict]:
        """Process multiple sections in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_section, section) 
                      for section in sections]
            return [future.result() for future in futures]
    
    def _process_single_section(self, section: Dict) -> Dict:
        """Process a single section (for parallel processing)"""
        try:
            # Add processing logic here based on section type
            return section
        except Exception as e:
            self.logger.error(f"Error processing section: {e}")
            return section
    
    def optimize_memory_usage(self):
        """Clear unnecessary data from memory"""
        if hasattr(self, 'processed_chunks'):
            del self.processed_chunks
        
        # Keep only recent embeddings
        if hasattr(self, 'embedding_cache'):
            cache_size = len(self.embedding_cache)
            if cache_size > self.max_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.embedding_cache.keys())[:cache_size//2]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                self.logger.info(f"Cleared {len(keys_to_remove)} old embeddings from cache")
        
        gc.collect()
    
    def _limit_section_title(self, title: str, max_words: int = 5) -> str:
        """Limit section title to maximum number of words"""
        if not title:
            return "Untitled"
        
        words = title.split()
        if len(words) <= max_words:
            # Return in title case without modification
            return title.title()
        
        # Limit words and format as title case without trailing dots
        return ' '.join(words[:max_words]).title()
    
    def _generate_contextual_title(self, content: str, original_title: str, persona: str, job: str) -> str:
        """Generate a contextual title using comprehensive content analysis"""
        if not content or len(content.strip()) < 20:
            return self._clean_title(original_title)
        
        # First, try to extract a complete sentence or phrase from content
        extracted_title = self._extract_meaningful_phrase(content)
        if extracted_title and len(extracted_title) > 10:
            return self._clean_title(extracted_title)
        
        # If original title is reasonable, enhance it with content context
        if original_title and len(original_title.strip()) > 5:
            enhanced_title = self._enhance_title_with_context(original_title, content)
            if enhanced_title:
                return self._clean_title(enhanced_title)
        
        # Fallback: create title from content keywords
        keyword_title = self._create_title_from_keywords(content)
        if keyword_title:
            return self._clean_title(keyword_title)
        
        # Final fallback
        return self._clean_title(original_title or "Document Section")
    
    def _extract_meaningful_phrase(self, content: str) -> str:
        """Extract a meaningful phrase or sentence from content that relates to the document"""
        # Clean content
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Look for patterns that indicate what the section teaches or explains
        teaching_patterns = [
            # Direct instructional content
            r'((?:How to|Learn to|Steps to|Ways to)\s+[^.]{10,70})',
            r'((?:Create|Build|Setup|Configure|Design|Use|Add|Edit|Manage)\s+[^.]{10,70}(?:forms?|documents?|fields?|signatures?|PDFs?))',
            # Feature descriptions
            r'([A-Z][^.]{15,70}(?:tool|feature|function|option|menu|button|command))',
            # Process explanations with PDF-related context
            r'([A-Z][^.]{15,70}(?:in Acrobat|in PDF|with forms|for documents))',
            # Acrobat-specific functionality
            r'((?:Fill & Sign|Prepare Forms|Digital Signatures?|Interactive Forms?|Form Fields?)[^.]{0,50})',
        ]
        
        for pattern in teaching_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                # Clean up the match
                candidate = self._clean_extracted_phrase(candidate)
                if 10 <= len(candidate) <= 80:
                    return candidate
        
        # Look for key concepts with document context
        concept_patterns = [
            r'((?:Creating|Building|Managing|Using|Setting up)\s+[^.]{10,60})',
            r'([A-Z][^.]{20,70}(?:workflow|process|procedure|method))',
            r'([A-Z][^.]{15,60}(?:allows you to|enables you to|helps you))',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content[:400], re.IGNORECASE)
            if matches:
                candidate = matches[0].strip()
                candidate = self._clean_extracted_phrase(candidate)
                if 15 <= len(candidate) <= 80:
                    return candidate
        
        # Look for the first substantial sentence that explains something
        sentences = re.split(r'[.!?]', content)
        for sentence in sentences[:4]:
            sentence = sentence.strip()
            if (20 <= len(sentence) <= 80 and 
                self._is_descriptive_sentence(sentence)):
                return self._clean_extracted_phrase(sentence)
        
        return None
    
    def _clean_extracted_phrase(self, phrase: str) -> str:
        """Clean an extracted phrase to make it a proper title"""
        # Remove leading articles and common prefixes
        phrase = re.sub(r'^(?:the|a|an)\s+', '', phrase, flags=re.IGNORECASE)
        
        # Remove trailing incomplete words
        phrase = re.sub(r'\s+(?:the|a|an|and|or|of|in|on|at|to|for|with)$', '', phrase, flags=re.IGNORECASE)
        
        # Clean up spacing
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        
        # Ensure it doesn't start with lowercase unless it's a proper noun
        if phrase and not phrase[0].isupper():
            phrase = phrase[0].upper() + phrase[1:]
        
        return phrase
    
    def _is_descriptive_sentence(self, sentence: str) -> bool:
        """Check if a sentence is descriptive and suitable for a title"""
        sentence_lower = sentence.lower()
        
        # Good indicators for title-worthy sentences
        good_indicators = [
            'acrobat', 'pdf', 'form', 'document', 'sign', 'fill', 'create', 
            'edit', 'export', 'share', 'field', 'signature', 'tool', 'feature'
        ]
        
        # Bad indicators (too generic or not informative)
        bad_indicators = [
            'this section', 'here you', 'the following', 'as shown', 
            'click here', 'see figure', 'refer to'
        ]
        
        # Check for good indicators
        has_good = any(indicator in sentence_lower for indicator in good_indicators)
        
        # Check for bad indicators
        has_bad = any(indicator in sentence_lower for indicator in bad_indicators)
        
        return has_good and not has_bad
    
    def _enhance_title_with_context(self, title: str, content: str) -> str:
        """Enhance an existing title with context from content to make it more document-relevant"""
        title = title.strip()
        content_lower = content.lower()
        
        # Check if we can identify what type of content this is
        content_type = self._identify_content_type(content)
        
        # If title is too generic, create a specific one based on content type
        generic_indicators = ['you can', 'section', 'part', 'introduction', 'overview']
        
        if any(indicator in title.lower() for indicator in generic_indicators) or len(title) < 15:
            # Create a specific title based on content analysis
            specific_title = self._create_specific_title(content, content_type)
            if specific_title and len(specific_title) > len(title):
                return specific_title
        
        # If title is already decent but could be enhanced
        if content_type and len(title) > 5:
            enhanced = self._add_context_to_title(title, content_type, content)
            if enhanced:
                return enhanced
        
        return title
    
    def _identify_content_type(self, content: str) -> str:
        """Identify what type of content this section covers"""
        content_lower = content.lower()
        
        # Map content indicators to types
        content_types = {
            'form_creation': ['prepare forms', 'create form', 'form field', 'interactive form', 'form design'],
            'fill_sign': ['fill & sign', 'fill and sign', 'filling forms', 'sign document', 'signature'],
            'editing': ['edit pdf', 'edit text', 'edit document', 'modify', 'change text'],
            'sharing': ['share pdf', 'send document', 'collaborate', 'review', 'comment'],
            'conversion': ['convert', 'create pdf', 'export', 'save as'],
            'digital_signatures': ['digital signature', 'certificate', 'secure sign', 'authentication'],
            'form_fields': ['text field', 'checkbox', 'dropdown', 'form element', 'field properties'],
            'validation': ['validate', 'verify', 'check', 'review form'],
        }
        
        for content_type, indicators in content_types.items():
            if any(indicator in content_lower for indicator in indicators):
                return content_type
        
        return 'general'
    
    def _create_specific_title(self, content: str, content_type: str) -> str:
        """Create a specific title based on content type and actual content"""
        
        # Templates for different content types
        title_templates = {
            'form_creation': 'Creating {} Forms',
            'fill_sign': 'Using Fill & Sign for {}',
            'editing': 'Editing {} in PDF',
            'sharing': 'Sharing and Collaborating on {}',
            'conversion': 'Converting {} to PDF',
            'digital_signatures': 'Digital Signatures for {}',
            'form_fields': 'Working with {} Form Fields',
            'validation': 'Validating {} Forms',
        }
        
        # Extract the specific object/context from content
        context_object = self._extract_context_object(content)
        
        if content_type in title_templates and context_object:
            return title_templates[content_type].format(context_object)
        elif content_type in title_templates:
            return title_templates[content_type].replace(' {}', '').replace('{}', 'Documents')
        
        # Fallback: extract key action and object
        action = self._extract_main_action(content)
        object_ref = self._extract_context_object(content)
        
        if action and object_ref:
            return f"{action} {object_ref}"
        elif action:
            return f"{action} in Acrobat"
        
        return None
    
    def _extract_context_object(self, content: str) -> str:
        """Extract what the content is about (forms, documents, etc.)"""
        content_lower = content.lower()
        
        # Common objects in Acrobat documentation
        objects = [
            'interactive forms', 'fillable forms', 'pdf forms', 'forms',
            'digital signatures', 'signatures', 'documents', 'pdfs',
            'form fields', 'text fields', 'checkboxes', 'dropdowns',
            'certificates', 'comments', 'annotations', 'markups'
        ]
        
        for obj in objects:
            if obj in content_lower:
                return obj.title()
        
        return 'Documents'
    
    def _extract_main_action(self, content: str) -> str:
        """Extract the main action being described"""
        content_lower = content.lower()
        
        actions = [
            'creating', 'building', 'designing', 'setting up',
            'filling', 'signing', 'completing',
            'editing', 'modifying', 'updating', 'changing',
            'sharing', 'sending', 'distributing',
            'converting', 'exporting', 'saving',
            'validating', 'verifying', 'reviewing'
        ]
        
        for action in actions:
            if action in content_lower:
                return action.title()
        
        return 'Working With'
    
    def _add_context_to_title(self, title: str, content_type: str, content: str) -> str:
        """Add relevant context to an existing title"""
        if content_type == 'general':
            return title
        
        # If title doesn't already contain context, add it
        context_words = ['form', 'pdf', 'document', 'signature', 'field']
        
        if not any(word in title.lower() for word in context_words):
            object_ref = self._extract_context_object(content)
            if object_ref and object_ref.lower() not in title.lower():
                return f"{title} - {object_ref}"
        
        return title
    
    def _create_title_from_keywords(self, content: str) -> str:
        """Create a title from the most important keywords in content"""
        # Extract meaningful words
        words = re.findall(r'\b[A-Za-z]{4,}\b', content[:400])
        if not words:
            return None
        
        # Filter and count word frequency
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower not in _STOPWORDS and len(word) > 3:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        if not word_freq:
            return None
        
        # Get top 3-4 most frequent meaningful words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:4]
        
        # Create a title from top words
        title_words = [word[0].title() for word in top_words]
        return ' '.join(title_words[:3])
    
    def _clean_title(self, title: str) -> str:
        """Clean and format a title properly"""
        if not title:
            return "Document Section"
        
        # Remove extra whitespace and clean
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Remove quotes if they wrap the entire title
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        
        # Remove common prefixes that make titles incomplete
        prefixes_to_remove = ['the ', 'a ', 'an ']
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix):
                title = title[len(prefix):]
        
        # Ensure title doesn't end with incomplete words
        if title.endswith(' the') or title.endswith(' a') or title.endswith(' an'):
            title = title.rsplit(' ', 1)[0]
        
        # Limit length
        if len(title) > 60:
            # Find a good breaking point
            if ' ' in title[50:60]:
                title = title[:title.rfind(' ', 50, 60)] + "..."
            else:
                title = title[:57] + "..."
        
        # Capitalize properly
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    def _improve_generic_title(self, title: str, content: str) -> str:
        """Improve generic titles using comprehensive content analysis"""
        if not content:
            return self._clean_title(title)
        
        # Use the same logic as contextual title generation
        return self._generate_contextual_title(content, title, "", "")
    
    def filter_relevant_content(self, documents: List[str], query: str, 
                              persona: str, job: str) -> List[str]:
        """Pre-filter documents based on relevance before expensive processing"""
        
        # Quick keyword matching
        job_keywords = self._get_job_keywords(job)
        persona_keywords = self._get_persona_keywords(persona)
        
        filtered_docs = []
        for doc in documents:
            doc_lower = doc.lower()
            relevance_score = 0
            
            # Score based on keyword presence
            for keyword in job_keywords + persona_keywords:
                if keyword.lower() in doc_lower:
                    relevance_score += 1
            
            # Only process documents with minimum relevance
            if relevance_score >= 2 or len(doc) > 500:  # Keep long docs
                filtered_docs.append(doc)
        
        # Limit based on processing mode
        max_docs = self.config.get('max_documents', 10)
        return filtered_docs[:max_docs]
    
    def analyze_section_relevance(self, section_content: str, section_title: str,
                                persona: str, job: str) -> Dict[str, Any]:
        """Analyze how relevant a section is to the persona and job"""
        if not section_content or len(section_content.strip()) < 10:
            return {
                "relevance_score": 3,
                "persona_match": 3,
                "job_match": 3,
                "explanation": "Insufficient content for analysis",
                "key_points": [],
                "contextual_title": self._improve_generic_title(section_title, section_content)
            }
        
        # Generate contextual title first
        contextual_title = self._generate_contextual_title(section_content, section_title, persona, job)
            
        prompt = f"""Analyze this section for {persona} planning {job}:

SECTION: {contextual_title}
ORIGINAL TITLE: {section_title}
CONTENT: {section_content[:800]}

Rate each aspect from 1-10:
RELEVANCE_SCORE: [how useful is this overall]
PERSONA_MATCH: [how well does this fit the persona's needs]
JOB_MATCH: [how directly does this help with the specific task]
EXPLANATION: [why this section is relevant - be specific]
KEY_POINTS: [specific actionable point 1] | [specific actionable point 2] | [specific actionable point 3]"""
        
        try:
            response = self.generate_response(prompt, temperature=0.1)
            
            if not response:
                return self._create_fallback_analysis(section_content, section_title, persona, job)
            
            # Parse the structured response
            analysis = self._parse_analysis_response(response)
            if analysis:
                analysis['contextual_title'] = contextual_title
                return analysis
            else:
                return self._create_fallback_analysis(section_content, contextual_title, persona, job)
                
        except Exception as e:
            self.logger.error(f"Section analysis failed: {str(e)}")
            return self._create_fallback_analysis(section_content, contextual_title, persona, job)
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse structured analysis response from Ollama with robust parsing"""
        try:
            lines = response.split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'relevance_score' in key:
                        analysis['relevance_score'] = self._extract_numeric_score(value)
                    elif 'persona_match' in key:
                        analysis['persona_match'] = self._extract_numeric_score(value)
                    elif 'job_match' in key:
                        analysis['job_match'] = self._extract_numeric_score(value)
                    elif 'explanation' in key:
                        analysis['explanation'] = value
                    elif 'key_points' in key:
                        analysis['key_points'] = [p.strip() for p in value.split('|') if p.strip()]
            
            # Validate required fields
            if all(k in analysis for k in ['relevance_score', 'persona_match', 'job_match']):
                return analysis
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to parse analysis response: {str(e)}")
            return None

    def _extract_numeric_score(self, value: str) -> int:
        """Extract numeric score from various formats like '7', '[6/10]', '7/10**', etc."""
        import re
        
        # Remove common formatting characters and extract first number
        # Patterns to match: 7, [6/10], 7/10**, [6/10]**, etc.
        numeric_pattern = r'(\d+)'
        matches = re.findall(numeric_pattern, value)
        
        if matches:
            score = int(matches[0])  # Take the first number found
            # Ensure score is in valid range (1-10)
            return max(1, min(10, score))
        else:
            # Fallback score if no number found
            return 5
    
    def _create_fallback_analysis(self, content: str, title: str, persona: str, job: str) -> Dict[str, Any]:
        """Create fallback analysis when Ollama fails"""
        # Simple keyword-based scoring
        persona_keywords = persona.lower().split()
        job_keywords = job.lower().split()
        content_lower = content.lower()
        title_lower = title.lower()
        
        persona_matches = sum(1 for kw in persona_keywords if kw in content_lower or kw in title_lower)
        job_matches = sum(1 for kw in job_keywords if kw in content_lower or kw in title_lower)
        
        # Calculate scores (3-8 range to avoid extremes)
        persona_score = min(8, max(3, 3 + persona_matches))
        job_score = min(8, max(3, 3 + job_matches))
        relevance_score = (persona_score + job_score) // 2
        
        return {
            "relevance_score": relevance_score,
            "persona_match": persona_score,
            "job_match": job_score,
            "explanation": f"Content analysis for {persona} regarding {job}. Section contains relevant information based on keyword matching.",
            "key_points": [
                f"Section title: {title[:50]}..." if len(title) > 50 else title,
                f"Content relevant to {persona}",
                f"Supports task: {job[:50]}..." if len(job) > 50 else job
            ],
            "contextual_title": title
        }
    
    def extract_refined_text(self, section_content: str, persona: str, job: str) -> str:
        """Extract and refine the most important parts of a section for any persona and job"""
        if not section_content or len(section_content.strip()) < 10:
            return f"As a {persona.lower()}, I don't have sufficient content available for your {job.lower()}."
        
        # Check cache first
        if self.cache_manager:
            cache_key = self.cache_manager.get_cache_key(section_content, persona, job, "extract_refined_text")
            cached_result = self.cache_manager.get_cached_response(cache_key)
            if cached_result:
                self.logger.debug("Using cached refined text")
                return self._clean_unicode_characters(cached_result)
            
        # Generic prompt that works for all personas and jobs
        prompt = f"""You are a {persona} helping with: {job}

Content to analyze:
{section_content}

Write a direct, professional response with specific actionable advice. Requirements:

STYLE:
- Use third-person perspective (The expert recommends, The specialist advises, etc.)
- Be direct and concise - avoid filler words
- No generic greetings or excitement ("Okay, let's plan!", "I'm thrilled!")
- No redundant introductions
- Professional tone throughout
- Avoid excessive enthusiasm or generic phrases
- Use only standard ASCII quotes and punctuation
- DO NOT wrap the response in quotation marks
- Start directly with the advice, no quotes around the entire response

CONTENT:
- limit the response to 60 maximum words not more than 60 words
- Specific actionable advice relevant to this job
- Concrete details that matter for this task
- Professional insights from your expertise as a {persona}
- Practical tips and recommendations
- All important points and details

START IMMEDIATELY with actionable advice. Example:
The {persona.lower()} recommends booking accommodations early because... 
NOT: "The {persona.lower()} recommends..." (no quotes around the whole response)

max 60 words response, no more than 60 words.

Response:"""
        
        try:
            refined = self.generate_response(prompt, temperature=0.1)
            
            # Clean up common generic responses
            if refined and not self._is_generic_response(refined):
                # Clean Unicode characters first
                refined = self._clean_unicode_characters(refined)
                # Convert to third-person perspective
                refined_clean = self._convert_to_third_person(refined.strip(), persona)
                # Frame with third-person intro if necessary
                refined_clean = self._ensure_first_person_perspective(refined_clean, persona, job)
                # Final Unicode cleanup
                refined_clean = self._clean_unicode_characters(refined_clean)
                
                # Cache the result
                if self.cache_manager:
                    self.cache_manager.cache_response(cache_key, refined_clean)
                
                return refined_clean
            else:
                self.logger.warning("Generic response detected, using enhanced fallback")
                fallback_result = self._create_enhanced_fallback_text(section_content, persona, job)
                fallback_result = self._clean_unicode_characters(fallback_result)
                
                # Cache the fallback result too
                if self.cache_manager:
                    self.cache_manager.cache_response(cache_key, fallback_result)
                
                return fallback_result
            
        except Exception as e:
            self.logger.error(f"Failed to refine text with Ollama: {str(e)}")
            fallback_result = self._create_enhanced_fallback_text(section_content, persona, job)
            fallback_result = self._clean_unicode_characters(fallback_result)
            
            # Cache the fallback result
            if self.cache_manager:
                self.cache_manager.cache_response(cache_key, fallback_result)
            
            return fallback_result
    
    def _is_generic_response(self, text: str) -> bool:
        """Check if response is too generic"""
        generic_patterns = [
            "okay, here's a summary",
            "here's a summary",
            "here is a summary",
            "summary for",
            "here's a plan",
            "here is a plan",
            "here's what you need to know",
            "here is what you need to know",
            "the content shows",
            "the document contains",
            "okay, let's plan",
            "let's plan this",
            "i'm excited",
            "i'm thrilled",
            "sounds fantastic",
            "going to be fantastic",
            "let's start by",
            "here's the breakdown"
        ]
        text_lower = text.lower()
        
        # Check for generic patterns
        has_generic_pattern = any(pattern in text_lower for pattern in generic_patterns)
        
        # Check if response is too short and generic
        is_too_short = len(text) < 100
        
        # Check for excessive excitement/enthusiasm
        excitement_words = ["fantastic", "excited", "thrilled", "amazing", "wonderful"]
        excitement_count = sum(1 for word in excitement_words if word in text_lower)
        
        return (has_generic_pattern and is_too_short) or excitement_count > 2

    def _ensure_first_person_perspective(self, text: str, persona: str, job: str) -> str:
        """Convert text to third-person perspective appropriate for the persona"""
        
        # Clean up any redundant intro phrases first
        text = self._clean_redundant_intros(text, persona)
        
        # Convert first-person indicators to third-person
        text = self._convert_to_third_person(text, persona)
        
        # Check if text needs third-person framing
        if text.startswith(("This", "The", "These", "It", "You", "When", "For")):
            words = text.split()
            if len(words) > 0:
                first_word = words[0].lower()
                if first_word in ["this", "the", "these", "it"]:
                    return f"The {persona.lower()} suggests that {text[len(words[0]):].strip()}"
                elif first_word == "you":
                    return f"The {persona.lower()} recommends that you{text[3:]}"
                elif first_word in ["when", "for"]:
                    return f"According to the {persona.lower()}, {text.lower()}"
        
        # For other cases, return the text as is if it's substantial content
        if len(text) > 50:
            return text
        
        # Only add intro for very short or generic responses
        return f"The {persona.lower()} advises: {text}"
    
    def _convert_to_third_person(self, text: str, persona: str) -> str:
        """Convert first-person phrases to third-person"""
        # Common first-person to third-person conversions
        conversions = {
            "i recommend": f"the {persona.lower()} recommends",
            "i suggest": f"the {persona.lower()} suggests",
            "i advise": f"the {persona.lower()} advises",
            "based on my": f"based on the {persona.lower()}'s",
            "in my experience": f"in the {persona.lower()}'s experience",
            "from my experience": f"from the {persona.lower()}'s experience",
            "i'd recommend": f"the {persona.lower()} would recommend",
            "i would suggest": f"the {persona.lower()} would suggest",
            "i think": f"the {persona.lower()} believes",
            "here's what i": f"here's what the {persona.lower()}",
            "let me": f"the {persona.lower()} will",
            "i'll": f"the {persona.lower()} will",
            "my advice": f"the {persona.lower()}'s advice"
        }
        
        text_lower = text.lower()
        result = text
        
        for first_person, third_person in conversions.items():
            if first_person in text_lower:
                # Find the actual case in the original text and replace
                import re
                pattern = re.compile(re.escape(first_person), re.IGNORECASE)
                result = pattern.sub(third_person, result)
        
        return result

    def _clean_redundant_intros(self, text: str, persona: str) -> str:
        """Remove redundant introductory phrases"""
        
        # Patterns to remove
        redundant_patterns = [
            f"As a {persona.lower()}, I recommend focusing on the following:",
            f"As a {persona.lower()}, I recommend focusing on these key points:",
            f"As a {persona.lower()}, I recommend",
            "focusing on the following:",
            "focusing on these key points:",
            "Okay, let's plan this",
            "Okay, let's",
            "I'm really excited about this",
            "I'm thrilled",
            "I'm confident we can",
            "Let's plan this",
            "Here's what I'd suggest:",
            "Here's the plan:"
        ]
        
        # Remove redundant patterns
        cleaned_text = text
        for pattern in redundant_patterns:
            if cleaned_text.lower().startswith(pattern.lower()):
                cleaned_text = cleaned_text[len(pattern):].strip()
                # Remove any leading punctuation or conjunctions
                cleaned_text = self._remove_leading_connectors(cleaned_text)
        
        # Remove multiple consecutive intro phrases
        while any(cleaned_text.lower().startswith(pattern.lower()) for pattern in redundant_patterns):
            for pattern in redundant_patterns:
                if cleaned_text.lower().startswith(pattern.lower()):
                    cleaned_text = cleaned_text[len(pattern):].strip()
                    cleaned_text = self._remove_leading_connectors(cleaned_text)
                    break
        
        return cleaned_text

    def _remove_leading_connectors(self, text: str) -> str:
        """Remove leading connectors and punctuation"""
        connectors = [
            "okay,", "well,", "so,", "now,", "first,", "let's", "here's", 
            "this", "that", "these", "those", "-", "—", "•", "*"
        ]
        
        text = text.strip()
        text_lower = text.lower()
        
        for connector in connectors:
            if text_lower.startswith(connector):
                text = text[len(connector):].strip()
                text_lower = text.lower()
        
        # Remove leading punctuation
        while text and text[0] in ",-–—•*":
            text = text[1:].strip()
            
        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        return text
    
    def _create_enhanced_fallback_text(self, content: str, persona: str, job: str) -> str:
        """Create enhanced fallback refined text with persona-specific first-person approach"""
        # Extract key sentences with actionable content
        sentences = content.replace('\n', ' ').split('.')
        relevant_sentences = []
        word_count = 0
        
        # Define persona-specific keywords for content relevance
        persona_keywords = self._get_persona_keywords(persona)
        job_keywords = self._get_job_keywords(job)
        
        # Look for sentences with specific information
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Prioritize actionable content and persona-relevant information
            has_specifics = any(indicator in sentence.lower() for indicator in [
                'recommend', 'suggest', 'tip', 'advice', 'should', 'must',
                'best', 'important', 'essential', 'key', 'critical',
                'effective', 'successful', 'proven', 'expert',
                'step', 'process', 'method', 'approach', 'strategy'
            ])
            
            has_persona_relevance = any(keyword in sentence.lower() for keyword in persona_keywords)
            has_job_relevance = any(keyword in sentence.lower() for keyword in job_keywords)
            
            if has_specifics or has_persona_relevance or has_job_relevance or len(relevant_sentences) < 10:
                relevant_sentences.append(sentence)
                word_count += len(sentence.split())
        
        if relevant_sentences:
            base_text = '. '.join(relevant_sentences) + '.'
            # Add persona-specific first-person introduction
            intro = f"As a {persona.lower()}, I recommend focusing on these key points for {job.lower()}: "
            result = intro + base_text
            return self._clean_unicode_characters(result)
        else:
            # Fallback with persona-specific advice using full content
            result = f"Based on my expertise as a {persona.lower()}, here are the key considerations for {job.lower()}: {content}"
            return self._clean_unicode_characters(result)

    def _get_persona_keywords(self, persona: str) -> List[str]:
        """Get relevant keywords based on persona type"""
        persona_lower = persona.lower()
        
        # Common professional keywords
        base_keywords = ['professional', 'expert', 'experience', 'knowledge', 'skill', 'best practice']
        
        if 'travel' in persona_lower or 'planner' in persona_lower:
            return base_keywords + ['destination', 'accommodation', 'transport', 'itinerary', 'booking', 'budget', 'activity']
        elif 'financial' in persona_lower or 'advisor' in persona_lower:
            return base_keywords + ['investment', 'budget', 'cost', 'expense', 'return', 'profit', 'analysis', 'risk']
        elif 'health' in persona_lower or 'medical' in persona_lower:
            return base_keywords + ['treatment', 'diagnosis', 'prevention', 'symptom', 'therapy', 'wellness', 'care']
        elif 'education' in persona_lower or 'teacher' in persona_lower:
            return base_keywords + ['learning', 'curriculum', 'assessment', 'student', 'objective', 'method', 'resource']
        elif 'business' in persona_lower or 'consultant' in persona_lower:
            return base_keywords + ['strategy', 'process', 'efficiency', 'growth', 'solution', 'implementation', 'result']
        elif 'legal' in persona_lower or 'lawyer' in persona_lower:
            return base_keywords + ['regulation', 'compliance', 'requirement', 'documentation', 'procedure', 'rights', 'obligation']
        elif 'tech' in persona_lower or 'developer' in persona_lower:
            return base_keywords + ['implementation', 'architecture', 'solution', 'framework', 'optimization', 'integration', 'system']
        else:
            return base_keywords + ['solution', 'approach', 'method', 'result', 'outcome', 'implementation']

    def _get_job_keywords(self, job: str) -> List[str]:
        """Get relevant keywords based on the job/task"""
        job_lower = job.lower()
        
        if 'plan' in job_lower:
            return ['schedule', 'timeline', 'organize', 'prepare', 'arrange', 'coordinate', 'structure']
        elif 'analy' in job_lower:  # analysis, analyze
            return ['evaluate', 'assess', 'examine', 'review', 'compare', 'investigate', 'research']
        elif 'develop' in job_lower or 'create' in job_lower:
            return ['design', 'build', 'construct', 'generate', 'produce', 'establish', 'formulate']
        elif 'manage' in job_lower:
            return ['oversee', 'control', 'supervise', 'monitor', 'coordinate', 'direct', 'administer']
        elif 'improve' in job_lower or 'optimize' in job_lower:
            return ['enhance', 'upgrade', 'refine', 'streamline', 'boost', 'increase', 'maximize']
        elif 'solve' in job_lower or 'fix' in job_lower:
            return ['resolve', 'address', 'correct', 'remedy', 'troubleshoot', 'repair', 'handle']
        else:
            return ['execute', 'implement', 'perform', 'accomplish', 'achieve', 'complete', 'deliver']
    
    def _create_fallback_refined_text(self, content: str, persona: str, job: str) -> str:
        """Create fallback refined text when Ollama fails - works for any persona/job"""
        # Simple extraction of first few sentences up to 200 words
        sentences = content.split('.')
        refined_sentences = []
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            refined_sentences.append(sentence)
            word_count += len(sentence.split())
        
        if refined_sentences:
            base_text = '. '.join(refined_sentences) + '.'
            # Add a persona-specific note about relevance
            summary = f" As a {persona.lower()}, I find this content relevant for {job.lower()}."
            result = base_text + summary
            return self._clean_unicode_characters(result)
        else:
            result = f"Based on my expertise as a {persona.lower()}, here are key insights from the document relevant to {job.lower()}. The content includes important details and actionable information for your specific needs."
            return self._clean_unicode_characters(result)
    
    def rank_sections(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Rank sections by importance for the given persona and job"""
        # Analyze each section
        analyzed_sections = []
        
        for section in sections:
            analysis = self.analyze_section_relevance(
                section['content'],
                section.get('section_title', ''),
                persona,
                job
            )
            
            section['analysis'] = analysis
            section['importance_score'] = analysis['relevance_score']
            analyzed_sections.append(section)
        
        # Sort by importance score
        analyzed_sections.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Assign ranks
        for i, section in enumerate(analyzed_sections):
            section['importance_rank'] = i + 1
        
        return analyzed_sections
    
    def generate_section_summary(self, sections: List[Dict], persona: str, job: str) -> str:
        """Generate a summary of selected sections with first-person perspective"""
        if not sections:
            return f"The {persona.lower()} doesn't have sufficient relevant sections available for {job.lower()}."
        
        sections_text = ""
        for section in sections[:5]:  # Top 5 sections
            # Use contextual title from analysis if available
            contextual_title = section.get('analysis', {}).get('contextual_title')
            section_title = contextual_title or section.get('section_title', 'Untitled')
            section_title = self._limit_section_title(section_title)
            sections_text += f"Section: {section_title}\n"
            sections_text += f"Content: {section['content']}\n\n"
        
        prompt = f"""You are a {persona} helping with: {job}

SECTIONS TO ANALYZE:
{sections_text}

Write a comprehensive third-person summary as if describing what a {persona.lower()} would advise on this task. Use "The {persona.lower()} recommends", "Based on the {persona.lower()}'s experience", "The {persona.lower()} suggests". Provide detailed insights and all important points for this specific job from the expertise of a {persona}.

Response:"""
        
        try:
            summary = self.generate_response(prompt)
            return self._ensure_first_person_perspective(summary, persona, job) if summary else f"The {persona.lower()} has analyzed the available sections for {job.lower()}."
        except Exception as e:
            self.logger.error(f"Failed to generate section summary: {str(e)}")
            return f"The {persona.lower()} has reviewed the available content for {job.lower()}. The sections contain relevant information for your specific needs."
    
    def _clean_unicode_characters(self, text: str) -> str:
        """Clean Unicode characters and replace with standard ASCII equivalents"""
        if not text:
            return text
        
        # Unicode character mappings to ASCII equivalents
        unicode_replacements = {
            # Quotation marks
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201a': "'",  # Single low-9 quotation mark
            '\u201e': '"',  # Double low-9 quotation mark
            
            # Dashes and hyphens
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2010': '-',  # Hyphen
            '\u2011': '-',  # Non-breaking hyphen
            
            # Spaces
            '\u00a0': ' ',  # Non-breaking space
            '\u2002': ' ',  # En space
            '\u2003': ' ',  # Em space
            '\u2009': ' ',  # Thin space
            
            # Other common Unicode characters
            '\u2026': '...',  # Horizontal ellipsis
            '\u00b7': '·',    # Middle dot
            '\u2022': '•',    # Bullet
            '\u00ae': '(R)',  # Registered trademark
            '\u00a9': '(C)',  # Copyright
            '\u2122': '(TM)', # Trademark
            
            # Accented characters (common ones)
            '\u00e9': 'e',    # é
            '\u00e8': 'e',    # è
            '\u00ea': 'e',    # ê
            '\u00eb': 'e',    # ë
            '\u00e1': 'a',    # á
            '\u00e0': 'a',    # à
            '\u00e2': 'a',    # â
            '\u00e4': 'a',    # ä
            '\u00ed': 'i',    # í
            '\u00ec': 'i',    # ì
            '\u00ee': 'i',    # î
            '\u00ef': 'i',    # ï
            '\u00f3': 'o',    # ó
            '\u00f2': 'o',    # ò
            '\u00f4': 'o',    # ô
            '\u00f6': 'o',    # ö
            '\u00fa': 'u',    # ú
            '\u00f9': 'u',    # ù
            '\u00fb': 'u',    # û
            '\u00fc': 'u',    # ü
            '\u00f1': 'n',    # ñ
            '\u00e7': 'c',    # ç
        }
        
        # Apply replacements
        cleaned_text = text
        for unicode_char, replacement in unicode_replacements.items():
            cleaned_text = cleaned_text.replace(unicode_char, replacement)
        
        # Remove any remaining problematic Unicode characters
        # Keep only printable ASCII characters, spaces, and common punctuation
        import re
        cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)
        
        # Remove wrapper quotes from the entire text
        cleaned_text = cleaned_text.strip()
        if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or \
           (cleaned_text.startswith("'") and cleaned_text.endswith("'")):
            cleaned_text = cleaned_text[1:-1].strip()
        
        return cleaned_text
