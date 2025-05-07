from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import os
import io
import zipfile
import json
from redis_conf import get_redis_client
import asyncio
from concurrent.futures import ProcessPoolExecutor

# from backend.redis_conf import get_redis_client
from settings import logger
from uuid import uuid4
from utils.utc_time import get_current_time_utc
from bson import ObjectId

from db import batches, candidates, jobs

app = FastAPI()


@app.post("/upload")
async def upload_pdf(
    job_id: str = Form(..., description="ID of the job to upload candidates for"),
    batch_name: str = Form(..., description="Name of the batch"),
    files: List[UploadFile] = File(...),
):
    try:

        current_dir: str = os.path.dirname(os.path.abspath(__file__))
        temp_dir: str = os.path.join(current_dir, "zip_dir")
        os.makedirs(temp_dir, exist_ok=True)

        processed_files: List = list()
        file_count: int = 0
        for file in files:
            # logger.info(f"Processing file: {file.filename},  {file}")
            if file.content_type not in [
                "application/zip",
                "application/x-zip-compressed",
            ]:
                logger.error(
                    f"Invalid file type for {file.filename}: {file.content_type}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not supported. Only ZIP files are allowed",
                )

            extracted_dir = os.path.join(temp_dir, file.filename.split(".", 1)[0])
            os.makedirs(extracted_dir, exist_ok=True)

            # contents: bytes = await file.read()
            # def unzip():
            #     with zipfile.ZipFile(io.BytesIO(contents)) as zip_file:
            #         zip_file.extractall(extracted_dir)
            # await asyncio.to_thread(unzip)

            # Process zip file
            contents: bytes = await file.read()
            with zipfile.ZipFile(io.BytesIO(contents)) as zip_file:
                zip_file.extractall(extracted_dir)

            processed_files.append(extracted_dir)

            curr_file_count = len(os.listdir(extracted_dir))
            file_count += curr_file_count
            logger.info(
                f"Extracted {curr_file_count} files from {file.filename}, Batch ID: "
            )

        data = {
            "job_id": job_id,
            "batch_name": batch_name,
            # "extracted_dir": processed_files,
            "upload_count": file_count,
            "company_id": "123_company",
            "batch_id": await generate_random_id(),
            "batch_name": batch_name,
            "upload_count": file_count,
            "job_id": await generate_obj_id(),
            "created_at": get_current_time_utc(),
            "updated_at": get_current_time_utc(),
        }

        await batches.insert_one(data)

        details: dict = dict()  # will have in actual app

        queue_data = {
            "extracted_dir": processed_files,
            "batch_id": data.get("batch_id"),
            "user_id": details.get("user_id", await generate_obj_id()),
            "company_id": details.get("company_id", await generate_obj_id()),
        }

        # print(processed_files)

        await enqueue(queue_data)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"{file_count} files queued for processing. You will receive an email when the process is done.",
                "batch_id": data.get("batch_id"),
            },
        )
    except Exception as e:
        # return HTTPException(
        #     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        #     detail=f"Something went wrong! {e}",
        # )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"Something went wrong! {e}",
                "batch_id": data.get("batch_id"),
            },
        )


async def generate_random_id() -> str:
    """Generates a random ID using UUID and a random number"""
    return str(uuid4())  # Generates a random UUID


async def generate_obj_id():
    return str(ObjectId())


async def enqueue(queue_data: dict):
    STREAM_NAME = "process_pdfs"
    redis_client = await get_redis_client()

    await redis_client.xadd(
        STREAM_NAME,
        # {"message": json.dumps(queue_data).encode()}
        {"message": json.dumps(queue_data).encode()},
    )
