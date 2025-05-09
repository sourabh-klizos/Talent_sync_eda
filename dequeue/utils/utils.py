from backend.settings import logger
import asyncio
import os
import shutil
from typing import List
import uuid
from dequeue.utils.utc_time import get_current_time_utc
from bson import Binary, ObjectId
from dequeue.parser.pdf_parser import extract, EmptyFileException

import os, sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.redis_conf import get_redis_client


from backend.db import extracted_texts, batches


async def delete_from_stream(stream_details: dict) -> None:
    stream_message_id = stream_details.get("stream_message_id")
    group_name = stream_details.get("group_name")
    stream_name = stream_details.get("stream_name")
    redis_client = await get_redis_client()
    await redis_client.xack(stream_name, group_name, stream_message_id)
    await redis_client.xdel(stream_name, stream_message_id)
    logger.info(f"Deleted message {stream_message_id} from stream '{stream_name}'")


# async def search_by_batch_id(
#     stream_name: str, target_batch_id: uuid.UUID
# ) -> str | None:
#     try:

#         redis_client = await get_redis_client()
#         entries = redis_client.xrange(stream_name)

#         for stream_id, data in entries:
#             if data and data.get("batch_id") == target_batch_id:
#                 logger.info(
#                     f"Found matching batch_id '{target_batch_id}' in stream '{stream_name}' at entry '{stream_id}'."
#                 )
#                 return stream_id

#         return None

#     except Exception as e:
#         logger.exception(
#             f"Error while searching for batch_id '{target_batch_id}' in stream '{stream_name}': {e}"
#         )
#         return None


# async def delete_task_from_queue(
#     stream_name: str, stream_id: str, target_batch_id: str
# ) -> bool:
#     try:

#         redis_client = await get_redis_client()
#         deleted_count = redis_client.xdel(stream_name, StopIteration)
#         if deleted_count == 1:
#             logger.info(
#                 f"successfully deleted queued task {stream_id} from stream {stream_name} batch_id {target_batch_id}"
#             )
#             return True
#         return False
#     except Exception as e:
#         logger.exception(
#             f"Error while deleting for batch_id '{target_batch_id}' in stream '{stream_name}' stream_id {stream_id} : {e}"
#         )
#         return False


# async def delete_queued_task(stream_name: str, target_batch_id: uuid.UUID) -> bool:
#     # redis_client = await get_redis_client()
#     stream_id = await search_by_batch_id(
#         stream_name=stream_name, target_batch_id=target_batch_id
#     )

#     if not stream_id:
#         return

#     is_completed = await check_task_already_completed(batch_id=target_batch_id)

#     if not is_completed:
#         is_deleted = await delete_task_from_queue(stream_id= stream_id,stream_name=stream_name,  target_batch_id=target_batch_id)
#         if  is_deleted:
#             return True

#     return False  # already completed


# from motor.motor_asyncio import AsyncIoMotorCollection

# async def check_task_already_completed(batch_id: str) -> bool:

#     upload_count = await batches.find_one({"batch_id":batch_id},{"upload_count" :1})
#     logger.info(f"Found {upload_count} uploads for batch_id {batch_id} ")

#     if upload_count:
#         curr = extracted_texts.find({"batch_id": batch_id})
#         extracted_texts_list = await curr.to_list(length=None)
#         length = len(extracted_texts_list)

#         if length == upload_count:
#             return True

#     return False


async def process_zip_extracted_files(
    extracted_dir: str,
    batch_id: uuid.UUID,
    job_id: str,
    user_id: str,
    company_id: str,
    stream_details: dict,
) -> None:
    """Process all PDF files in the extracted directory using worker pool"""
    logger.info(f"Starting to process files from extracted directory: {extracted_dir}")
    try:
        files = [f for f in os.listdir(extracted_dir) if f.endswith((".pdf", ".docx"))]
        logger.info(f"Found {len(files)} CV files to process")

        logger.info(f"files {files}")

        # Split files into non-overlapping chunks of size 4
        chunks = [files[i : i + 4] for i in range(0, len(files), 4)]
        logger.info(f"Split files into {len(chunks)} non-overlapping chunks")

        # Fetch job data once
        # job_data = redis.get_json_(f"job:{job_id}")
        job_data: dict = dict()

        # Process chunks with 8 concurrent workers
        semaphore = asyncio.Semaphore(8)

        async def process_with_semaphore(chunk):
            nonlocal batch_id, job_id, job_data
            async with semaphore:
                await _process_file_chunks(
                    chunk,
                    extracted_dir,
                    batch_id,
                    job_id,
                    job_data,
                    user_id,
                    company_id,
                )

        # chunk_tasks = [asyncio.create_task(process_with_semaphore(chunk)) for chunk in chunks]
        chunk_tasks = [process_with_semaphore(chunk) for chunk in chunks]
        await asyncio.gather(*chunk_tasks)

        logger.info("Completed processing all chunks")

        # Process qualified candidates and prepare email notifications

        logger.info(f"Starting processing for batch {batch_id}")
        await asyncio.sleep(2)
        logger.info(f"Successfully processed batch {batch_id}")
        # asyncio.create_task(process_candidates_and_vectorize(batch_id, job_id, company_id, user_id))

    finally:
        # Cleanup extracted directory
        try:
            # shutil.rmtree(os.path.dirname(extracted_dir))
            shutil.rmtree(extracted_dir)
            logger.info(f"Successfully cleaned up directory: {extracted_dir}")

            await delete_from_stream(stream_details)
            logger.info(f"Successfully cleaned up directory: {extracted_dir}")

        except Exception as e:
            logger.error(
                f"Failed to cleanup directory {extracted_dir}: {str(e)}", exc_info=True
            )


async def _process_file_chunks(
    chunks: List[str],
    extracted_dir: str,
    batch_id: uuid.UUID,
    job_id: str,
    job_data: dict,
    user_id: str,
    company_id: str,
):
    """Process a chunk of PDF files concurrently and efficiently"""
    logger.info(f"Processing chunk of {len(chunks)} files from {extracted_dir}")

    # Process PDFs concurrently
    tasks = [
        _process_files(
            os.path.join(extracted_dir, file), job_data.get("job_id"), user_id, batch_id
        )
        for file in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Efficiently separate valid results and count errors
    valid_results, invalid_results = [], []
    error_count = 0

    # Get current time
    current_time = get_current_time_utc()

    for result in results:
        if isinstance(result, Exception):
            error_count += 1
            logger.error(f"Failed to process PDF: {str(result)}", exc_info=True)
        else:
            if isinstance(result, dict) and result.get("error"):
                invalid_results.append(
                    {
                        **result,
                        "batch_id": batch_id,
                        "job_id": ObjectId(job_id),
                        "company_id": ObjectId(company_id),
                        "updated_at": current_time,
                        "created_at": current_time,
                    }
                )
            # elif isinstance(result, CVParseResponse):
            elif isinstance(result, list):
                valid_results.append(result)

    logger.info(
        f"Processed chunk: {len(valid_results)} successful, {error_count} failed"
    )

    await asyncio.sleep(2)

    # Insert valid results into MongoDB
    # if valid_results:
    #     try:
    #         # Build candidates data efficiently
    #         candidates_data = [
    #             {
    #                 **result.model_dump(),
    #                 "batch_id": Binary.from_uuid(batch_id),
    #                 "job_id": ObjectId(job_id),
    #                 "company_id": ObjectId(company_id),
    #                 "updated_at": current_time,
    #                 "created_at": current_time,
    #                 "status": "New",
    #                 "compatibility_analysis": await compatibility_analyzer.get_analysis(job=job_data, candidate=result.model_dump(), user_id=user_id),
    #             }
    #             for result in valid_results
    #         ]

    #         # Track emails we've already seen, prevents unnecessary database writes if the chunk has same PDFs.
    #         processed_candidates = set()
    #         operations = []

    #         for candidate in candidates_data:
    #             email = candidate.get("email")
    #             name = candidate.get("name")

    #             # Create a unique identifier using both name and email
    #             candidate_key = f"{name}:{email}" if name and email else None

    #             if candidate_key and candidate_key not in processed_candidates:

    #                 operations.append(
    #                     ReplaceOne(
    #                         {
    #                             "email": email,
    #                             "name": name,
    #                             "job_id": candidate.get("job_id"),
    #                             "company_id": candidate.get("company_id"),
    #                         },
    #                         candidate,
    #                         upsert=True,
    #                     )
    #                 )
    #                 # Mark this candidate as processed
    #                 processed_candidates.add(candidate_key)
    #             else:
    #                 # Adding logs when skipping a candidate
    #                 if not name:
    #                     invalid_results.append(candidate)
    #                     logger.debug(f"skipped candidate with no name, Backed up details: {candidate}")
    #                 else:
    #                     logger.info(f"Found same candidate in batch, email: {email}, skipping...")
    #         # Execute bulk operation if we have any operations
    #         if operations:
    #             # result = await candidates.bulk_write(operations, ordered=False)
    #             logger.info(f"Successfully inserted {len(operations)} candidates into database")

    #     except Exception as e:
    #         logger.error(f"Failed to insert candidates into database: {str(e)}", exc_info=True)

    # if invalid_results:
    #     # await candidates_errors.insert_many(invalid_results, ordered=False)
    #     logger.info(f"Successfully inserted {len(invalid_results)} errored candidate details into database")


class TextExtractionFailedException(Exception):
    pass


async def _process_files(file_path: str, job_name: str, user_id: str, batch_id: str):
    """Process a single PDF file and return parsed CV data"""
    logger.info(f"Starting to process PDF file: {file_path}")

    # unq_id = _generate_unique_candidate_id()
    unq_id = uuid.uuid4()

    try:
        text, is_image_pdf = await extract.extract_text(file_path, user_id)

        if text:
            extracted_texts.insert_one(
                {
                    "job_name": job_name,
                    "user_id": user_id,
                    "batch_id": batch_id,
                    "parse_text": text if text else "",
                    "is_image_pdf":is_image_pdf
                }
            )

        logger.info(f"text extracted ... for  {file_path}")
        if file_path.lower().endswith((".docx", ".doc")) and not text:
            # raise exception on failed word files
            raise TextExtractionFailedException("Error parsing")

        await asyncio.sleep(2)

        # Parse the CV
        # for image pdfs the text is already coming with the Json Format
        # parsed_cv = await cv_parser.parse_text(text, user_id) if not is_image_pdf else text

        # if parsed_cv.email:
        #     existing_unique_id = await candidates.find_one({"email": parsed_cv.email}, {"_id": 0, "unique_id": 1})
        #     if existing_unique_id and existing_unique_id.get("unique_id"):
        #         unq_id = existing_unique_id.get("unique_id")

        # Set the unique ID for the candidate
        # parsed_cv.unique_id = unq_id

        # Upload the CV to S3, set directory link
        # parsed_cv.cv_directory_link = upload_file_to_s3(file_path, job_name, unq_id)s

        logger.info(
            f"Successfully parsed {'PDF' if file_path.endswith('pdf') else 'WORD'} file: {file_path}"
        )

        # return parsed_cv
        return text

    except Exception as e:
        if isinstance(e, EmptyFileException):
            logger.error(
                f"{file_path}, contains no text. Moving to errors collection. \n{str(e)}",
                exc_info=True,
            )
        else:
            logger.error(
                f"Error extracting text from {file_path}: {str(e)}, moving to errors collection",
                exc_info=True,
            )
        # return {"error": str(e), "cv_directory_link": upload_file_to_s3(file_path, job_name, unq_id)}
        return {"error": str(e), "cv_directory_link": "link"}
