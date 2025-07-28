"""
Cache management system for improved performance
"""
import hashlib
import json
import os
import time
from typing import Dict, Optional, Any
import logging

class CacheManager:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Clean expired cache on startup
        self._cleanup_expired_cache()
    
    def get_cache_key(self, content: str, persona: str, job: str, method: str = "default") -> str:
        """Generate cache key from content and parameters"""
        data = f"{method}:{content}:{persona}:{job}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached response if exists and not expired"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            if time.time() - cached_data.get('timestamp', 0) > self.ttl:
                os.remove(cache_file)
                return None
            
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return cached_data['response']
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self.logger.warning(f"Error reading cache file {cache_file}: {e}")
            try:
                os.remove(cache_file)
            except OSError:
                pass
            return None
    
    def cache_response(self, cache_key: str, response: Any):
        """Cache the response with timestamp"""
        try:
            # Check cache size limit
            self._enforce_cache_limit()
            
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            cache_data = {
                'response': response,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
    
    def _cleanup_expired_cache(self):
        """Remove expired cache files"""
        try:
            current_time = time.time()
            removed_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    if current_time - cached_data.get('timestamp', 0) > self.ttl:
                        os.remove(file_path)
                        removed_count += 1
                        
                except (json.JSONDecodeError, KeyError, OSError):
                    # Remove corrupted cache files
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except OSError:
                        pass
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired cache files")
                
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    def _enforce_cache_limit(self):
        """Enforce maximum cache size by removing oldest files"""
        try:
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    stat = os.stat(file_path)
                    cache_files.append((file_path, stat.st_mtime))
            
            if len(cache_files) >= self.max_size:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x[1])
                
                # Remove oldest files to make room
                files_to_remove = len(cache_files) - self.max_size + 1
                for file_path, _ in cache_files[:files_to_remove]:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                
                self.logger.info(f"Removed {files_to_remove} old cache files to enforce size limit")
                
        except Exception as e:
            self.logger.error(f"Error enforcing cache limit: {e}")
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': self.cache_dir,
                'max_size': self.max_size,
                'ttl_hours': self.ttl / 3600
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
