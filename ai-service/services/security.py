import re

MAX_INPUT_LENGTH = 1000

# Prompt injection patterns
PROMPT_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+all\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(system|hidden|secret)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
]

# HTML tag removal pattern
HTML_PATTERN = re.compile(r"<[^>]+>")

# Security functions for input validation and sanitization
def detect_prompt_injection(text: str) -> bool:
    # Check if any prompt injection pattern matches the input text
    return any(pattern.search(text) for pattern in PROMPT_PATTERNS)


def sanitize(text: str) -> str:
    # Remove HTML tags and trim whitespace
    return re.sub(HTML_PATTERN, "", text).strip()

# Input validation function that checks type, length, and prompt injection
def validate_input(text: str):
    # Basic type and length checks
    if not isinstance(text, str):
        return False, "Input must be a string"
    # Check for empty input after trimming
    if not text.strip():
        return False, "Input cannot be empty"
    # Check for excessively long input
    if len(text) > MAX_INPUT_LENGTH:
        return False, "Input too long (max 1000 chars)"
    # Check for prompt injection patterns
    if detect_prompt_injection(text):
        return False, "Prompt injection detected"

    return True, None