import asyncio
import redis.asyncio as redis
import json
import logging
from typing import Dict
import sys, os
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from backend.redis_conf import get_redis_client
from redis_conf import get_redis_client
from backend.settings import logger
from utils.utils import process_zip_extracted_files

REDIS_URI = "redis://localhost"
STREAM_NAME = "process_pdfs"
GROUP_NAME = "tread_group"
MIN_IDLE_TIME_MS = 60 * 1_000 * 3  # Reclaim jobs stuck for 180 seconds
READ_BLOCK_MS = 5000
READ_COUNT = 10


async def ensure_group(redis_client):
    try:
        await redis_client.xgroup_create(
            STREAM_NAME, GROUP_NAME, id="0-0", mkstream=True
        )
        logger.info("Consumer group created")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group already exists")
        else:
            raise


async def process_message(message_id: str, data: str):
    logger.info(f"Processing message {message_id} -> {data}")
    for extracted_dir in data.get("message", {}).get("extracted_dir", []):
        logger.info(f"Processing extracted directory: {extracted_dir}")
        await process_zip_extracted_files(
            extracted_dir,
            data.get("batch_id"),
            data.get("job_id"),
            data.get("company_id"),
            data.get("user_id"),
        )
    logger.info(f"Done processing {message_id}")


async def consume_new_messages(redis_client, consumer_name):
    while True:
        try:
            response = await redis_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername=consumer_name,
                streams={STREAM_NAME: ">"},
                count=READ_COUNT,
                block=READ_BLOCK_MS,
            )
            if not response:
                continue

            for stream, messages in response:
                for message_id, message in messages:
                    try:
                        data = {k: json.loads(v) for k, v in message.items()}
                        logger.info(f"Response: {data}")
                        await process_message(message_id, data)
                        await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Error decoding JSON for message {message_id}: {e}"
                        )
                    except Exception as e:
                        logger.error(f"Unexpected error for message {message_id}: {e}")
        except Exception as e:
            logger.exception(f"Error while reading new messages: {e}")


async def reclaim_stuck_messages(redis_client, consumer_name):
    while True:
        try:
            messages = await redis_client.xautoclaim(
                STREAM_NAME,
                GROUP_NAME,
                consumer_name,
                min_idle_time=MIN_IDLE_TIME_MS,
                start_id="0-0",
                count=READ_COUNT,
            )
            if not messages or not messages[1]:
                await asyncio.sleep(10)
                continue

            for message_id, message in messages[1]:
                logger.warning(
                    f"Reclaiming and reprocessing stuck message: {message_id}"
                )
                try:
                    data = {k: json.loads(v) for k, v in message.items()}
                    await process_message(message_id, data)
                    await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON for message {message_id}")
        except Exception as e:
            logger.exception(f"Error while reclaiming messages: {e}")
        await asyncio.sleep(5)


async def run_consumer(consumer_name):
    redis_client = await get_redis_client()
    await ensure_group(redis_client)
    await asyncio.gather(
        consume_new_messages(redis_client, consumer_name),
        # reclaim_stuck_messages(redis_client, consumer_name)
    )


def start_consumer_process(consumer_name):
    asyncio.run(run_consumer(consumer_name))


if __name__ == "__main__":
    # consumer_names = ["consumer-1", "consumer-2", "consumer-3"]
    consumer_names = ["consumer-10", "consumer-20"]
    processes = []

    for name in consumer_names:
        p = multiprocessing.Process(target=start_consumer_process, args=(name,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
