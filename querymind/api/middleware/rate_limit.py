import time
from collections import defaultdict
from fastapi import HTTPException, Request, status

class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > window_start
        ]
        
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        
        self._requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        recent = [ts for ts in self._requests[client_id] if ts > window_start]
        return max(0, self.max_requests - len(recent))


# Global instance
limiter = RateLimiter(max_requests=20, window_seconds=60)


async def rate_limit_dependency(request: Request):
    client_ip = request.client.host
    
    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 20 requests per minute.",
            headers={"Retry-After": "60"},
        )
