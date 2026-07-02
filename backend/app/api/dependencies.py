import secrets
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import get_settings, Settings

# Define the expected header key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(
    api_key: str = Security(api_key_header),
    settings: Settings = Depends(get_settings)
) -> str:
    """
    Validates the API key from the request header against the configured secret.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in headers",
        )
    
    # compare_digest prevents attackers from guessing the key via response time measurements
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
        
    return api_key