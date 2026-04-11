from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from ..core.config import settings

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    database = None

db = MongoDB()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.mongodb_url)
    db.database = db.client[settings.database_name]
    print("Connected to MongoDB")

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Disconnected from MongoDB")

def get_database():
    return db.database