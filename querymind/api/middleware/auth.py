import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected_key = os.getenv("QUERYMIND_API_KEY")
    
    if not expected_key:
        raise RuntimeError("QUERYMIND_API_KEY environment variable not set")
    
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return api_key
