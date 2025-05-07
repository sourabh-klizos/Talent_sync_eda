
# from backend.redis_conf import get_redis_client
from redis_conf import get_redis_client
from settings import logger
import uuid
from db import extracted_texts, batches
import json
import os, shutil
from typing import List

async def search_by_batch_id(
    stream_name: str, target_batch_id: uuid.UUID
) -> str | None:
    try:

        redis_client = await get_redis_client()
        entries = await redis_client.xrange(stream_name)

        if not entries:
            return None , None

        # print(f"stream data {entries}")

        for stream_id, data in entries:
            data = json.loads(data.get("message"))
        
            # print(data.get("message"))
            # print(stream_id)
            if data and data.get("batch_id") == target_batch_id:
                logger.info(
                    f"Found matching batch_id '{target_batch_id}' in stream '{stream_name}' at entry '{stream_id}'."
                )
                return stream_id , data.get("extracted_dir")

        return None , None

    except Exception as e:
        logger.exception(
            f"Error while searching for batch_id '{target_batch_id}' in stream '{stream_name}': {e}"
        )
        return None , None




async def cleanup_cancelled_task_dirs(dirs: List[str]):
    for dir_path in dirs:
        print(dir_path)
        if os.path.exists(dir_path):
            try:
                logger.info(f"Deleting directory: {dir_path}")
                shutil.rmtree(dir_path) 
                logger.info(f"Successfully deleted: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to delete {dir_path}: {e}")
        else:
            logger.warning(f"Directory not found: {dir_path}")

    


async def delete_task_from_queue(
    stream_name: str, stream_id: str, target_batch_id: str
) -> bool:
    try:

        redis_client = await get_redis_client()
        deleted_count = await redis_client.xdel(stream_name, stream_id)
        if deleted_count == 1:
            # TODO Cleam Directories After Delete 

            logger.info(
                f"successfully deleted queued task {stream_id} from stream {stream_name} batch_id {target_batch_id}"
            )
            return True
        return False
    except Exception as e:
        logger.exception(
            f"Error while deleting for batch_id '{target_batch_id}' in stream '{stream_name}' stream_id {stream_id} : {e}"
        )
        return False


async def delete_queued_task(stream_name: str, target_batch_id: uuid.UUID) -> bool:
    # redis_client = await get_redis_client()
    # print(stream_name, target_batch_id)
    stream_id , extraxted_dir_list = await search_by_batch_id(
        stream_name=stream_name, target_batch_id=target_batch_id
    )
    print(f"stream_id {stream_id}")

    if not stream_id:
        return
    
    # is_completed = await check_task_already_completed(batch_id=target_batch_id)

    # if not is_completed:
    #     is_deleted = await delete_task_from_queue(stream_id= stream_id,stream_name=stream_name,  target_batch_id=target_batch_id )
    #     if  is_deleted:
    #         await cleanup_cancelled_task_dirs(extraxted_dir_list)

    #         return True

    # is_completed = await check_task_already_completed(batch_id=target_batch_id)

    # if not is_completed:
    is_deleted = await delete_task_from_queue(stream_id= stream_id,stream_name=stream_name,  target_batch_id=target_batch_id )
    if  is_deleted:
        await cleanup_cancelled_task_dirs(extraxted_dir_list)

        return True
        
    return False  # already completed


    


# from motor.motor_asyncio import AsyncIoMotorCollection

async def check_task_already_completed(batch_id: str) -> bool:

    upload_count = await batches.find_one({"batch_id":batch_id},{"upload_count" :1, "_id":0})
    logger.info(f"Found {upload_count} uploads for batch_id {batch_id} ")

    

    if upload_count:
        logger.info(f"Found upload_count: {upload_count} for batch_id: {batch_id}")

        curr = extracted_texts.find({"batch_id": batch_id})
        extracted_texts_list = await curr.to_list(length=None)
        length = len(extracted_texts_list)

        if length == upload_count.get("upload_count"):
            logger.info(f"All {upload_count} files processed for batch_id: {batch_id}. Task is complete.")
            return True
        else:
            logger.info(f"Only {length}/{upload_count} files processed for batch_id: {batch_id}. Task not complete.")
        
    return False
