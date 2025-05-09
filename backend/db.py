import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorCollection
from backend.settings import settings


# MongoDB connection setup
MONGO_URI = settings.mongo_uri
DATABASE_NAME = settings.database_name


# MONGO_URI = "mongodb://localhost:27017/"
# DATABASE_NAME = "talentsync_"


BATCHES_COLLECTION = "batches"
CANDIDATES_COLLECTION = "candidates"
JOB_COLLECTION = "jobs"
EXTRACTED_TEXT_COLLECTION = "extracted_texts"

# Create an async MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

# Select the database
db = client[DATABASE_NAME]

# Select collections
# batches: AsyncIOMotorCollection = db[BATCHES_COLLECTION]
# candidates: AsyncIOMotorCollection = db[CANDIDATES_COLLECTION]
# jobs: AsyncIOMotorCollection = db[JOB_COLLECTION]

batches = db[BATCHES_COLLECTION]
candidates = db[CANDIDATES_COLLECTION]
jobs = db[JOB_COLLECTION]
extracted_texts = db[EXTRACTED_TEXT_COLLECTION]
