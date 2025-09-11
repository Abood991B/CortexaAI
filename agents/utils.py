"""Common utility functions for the Multi-Agent Prompt Engineering System.

This module contains shared functionality to reduce code duplication across agents.
"""

from typing import List, Dict, Any
import logging
from config.config import get_logger

logger = get_logger(__name__)


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable based on its characteristics.
    
    This function is used by multiple agents to determine if they should
    retry after encountering an error.
    
    Args:
        error: The exception to check
        
    Returns:
        bool: True if the error is retryable, False otherwise
    """
    if not error:
        return False
        
    error_str = str(error).lower()
    retryable_indicators = [
        "rate limit", "timeout", "connection", "network",
        "temporary", "server error", "502", "503", "504",
        "internal server error", "service unavailable"
    ]
    
    # Check if any retryable indicator is in the error message
    is_retryable = any(indicator in error_str for indicator in retryable_indicators)
    
    logger.debug(f"Error retryability check: {is_retryable} for error: {error_str[:100]}")
    return is_retryable


def sanitize_json_response(raw_output: Any) -> str:
    """Sanitize raw LLM output to extract clean JSON.
    
    Handles cases where JSON is wrapped in markdown code fences or
    contains other formatting issues.
    
    Args:
        raw_output: Raw output from LLM
        
    Returns:
        str: Clean JSON string
    """
    import json
    import re
    
    try:
        # If the output is already a dict, dump it to a string
        if isinstance(raw_output, dict):
            return json.dumps(raw_output)
        
        # Extract content from response object
        if hasattr(raw_output, 'content'):
            content = raw_output.content
        else:
            content = str(raw_output)
        
        # Use regex to find content within ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if match:
            clean_json = match.group(1).strip()
        else:
            # Fallback: find JSON object boundaries
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                clean_json = content[start_idx:end_idx+1].strip()
            else:
                clean_json = content
        
        # Fix common JSON errors from LLMs
        clean_json = clean_json.replace("\\'", "'")
        
        # Validate JSON
        json.loads(clean_json)
        return clean_json
        
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.error(f"Failed to sanitize JSON output. Error: {e}")
        return '{}'  # Return empty object as fallback


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts.
    
    Simple word overlap based similarity for basic comparisons.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
        
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def extract_key_topics(text: str, max_topics: int = 10) -> List[str]:
    """Extract key topics/keywords from text.
    
    Simple keyword extraction based on frequency and importance.
    
    Args:
        text: Input text
        max_topics: Maximum number of topics to extract
        
    Returns:
        List of key topics
    """
    import re
    from collections import Counter
    
    # Common stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Filter out stop words and short words
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Count frequencies
    word_counts = Counter(meaningful_words)
    
    # Get most common topics
    topics = [word for word, _ in word_counts.most_common(max_topics)]
    
    return topics


def format_context_parts(context_parts: List[Dict[str, Any]]) -> str:
    """Format context parts for inclusion in prompts.
    
    Args:
        context_parts: List of context dictionaries
        
    Returns:
        Formatted string for prompt inclusion
    """
    if not context_parts:
        return "No additional context available."
    
    formatted_parts = []
    for i, part in enumerate(context_parts, 1):
        relevance = part.get('relevance', 0)
        formatted_parts.append(f"""
[{i}] {part.get('type', 'UNKNOWN').upper()} CONTEXT (Relevance: {relevance:.2f})
{part.get('content', '')}
""")
    
    return "\n".join(formatted_parts)


def validate_api_response(response: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that an API response contains all required fields.
    
    Args:
        response: Response dictionary to validate
        required_fields: List of field names that must be present
        
    Returns:
        bool: True if all required fields are present
    """
    if not isinstance(response, dict):
        return False
        
    for field in required_fields:
        if field not in response:
            logger.warning(f"Missing required field in response: {field}")
            return False
            
    return True


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length] + suffix


def merge_dicts_deep(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
            
    return result
