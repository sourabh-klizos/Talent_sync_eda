# import asyncio
# import redis.asyncio as redis

# _redis_client = None
# _redis_lock = asyncio.Lock()

# REDIS_URI = "redis://localhost:6379/0"


# async def get_redis_client(max_connections: int = 10) -> redis.Redis:
#     global _redis_client
#     if _redis_client is None:
#         async with _redis_lock:
#             if _redis_client is None:  # Double-check inside the lock
#                 pool = redis.ConnectionPool.from_url(
#                     REDIS_URI, max_connections=max_connections, decode_responses=True
#                 )
#                 _redis_client = redis.Redis(connection_pool=pool)
#     return _redis_client


import redis.asyncio as redis
# from settings import settings
_redis_client = None

from dotenv import load_dotenv
import os

load_dotenv()

REDIS_URI = os.getenv("REDIS_URL")
# REDIS_URI = "redis://localhost:6379/0"
# REDIS_URI = settings.redis_url


async def get_redis_client(max_connections: int = 100) -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        pool = redis.ConnectionPool.from_url(
            REDIS_URI, max_connections=max_connections, decode_responses=True
        )
        _redis_client = redis.Redis(connection_pool=pool)
    return _redis_client
