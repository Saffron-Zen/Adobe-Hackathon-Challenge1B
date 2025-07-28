"""
Optimized configuration settings for performance improvements
"""

class OptimizedConfig:
    # Model settings
    FAST_MODEL = "gemma3:1b"  # Current
    BALANCED_MODEL = "llama3.2:3b"  # Better quality
    QUALITY_MODEL = "llama3.2:8b"  # Best quality
    
    # Processing settings
    MAX_CHUNK_SIZE = 512  # Optimal for embedding models
    OVERLAP_SIZE = 50     # Reduced overlap for speed
    MAX_WORKERS = 4       # Parallel processing
    BATCH_SIZE = 32       # Embedding batch size
    
    # Cache settings
    ENABLE_CACHE = True
    CACHE_TTL = 3600      # 1 hour cache expiry
    MAX_CACHE_SIZE = 1000 # Maximum cached items
    
    # Quality vs Speed trade-offs
    QUICK_MODE = {
        "max_documents": 5,
        "max_sections": 3,
        "temperature": 0.3,
        "max_workers": 2
    }
    
    BALANCED_MODE = {
        "max_documents": 10,
        "max_sections": 5, 
        "temperature": 0.1,
        "max_workers": 4
    }
    
    QUALITY_MODE = {
        "max_documents": 20,
        "max_sections": 10,
        "temperature": 0.05,
        "max_workers": 8
    }

def determine_processing_mode(file_size: int, urgency: str = "normal") -> str:
    """Automatically determine best processing mode"""
    if urgency == "urgent" or file_size < 1_000_000:  # < 1MB
        return "quick"
    elif file_size < 10_000_000:  # < 10MB
        return "balanced" 
    else:
        return "quality"

def get_mode_config(mode: str) -> dict:
    """Get configuration for specified mode"""
    return getattr(OptimizedConfig, f"{mode.upper()}_MODE", OptimizedConfig.BALANCED_MODE)
