import asyncio
import redis.asyncio as redis
import json
import logging
from typing import Dict
from redis_conf import get_redis_client

from settings import logger

REDIS_URI = "redis://localhost"
STREAM_NAME = "process_pdfs"
GROUP_NAME = "tread_group"
MIN_IDLE_TIME_MS = 60 * 1_000 * 2  # Reclaim jobs stuck for 120 seconds
READ_BLOCK_MS = 5000  # Block 5 seconds for new messages
READ_COUNT = 10


async def ensure_group(redis_client):
    try:
        await redis_client.xgroup_create(STREAM_NAME, GROUP_NAME, id='0-0', mkstream=True)
        logger.info("Consumer group created")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group already exists")
        else:
            raise


async def process_message(message_id: str, data: Dict):
    #This will receive list of path dir
    logger.info(f"Processing message {message_id} -> {data}")
    # Simulate async work (like PDF parsing)
    await asyncio.sleep(10)

    #this function will receive multipe paths of pdf extraxt dir 
    # background_processing()
    logger.info(f"Done processing {message_id}")


async def consume_new_messages(redis_client, consumer_name):
    while True:
        try:
            response = await redis_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername=consumer_name,
                streams={STREAM_NAME: '>'},
                count=READ_COUNT,
                block=READ_BLOCK_MS
            )
            if not response:
                logger.info("No response")
                continue

            for stream, messages in response:
                # print(stream, " ----" *10)
                # print(messages, " ----" *10)
                for message_id, message in messages:
                    try:
                        # data = {k.decode(): json.loads(v) for k, v in message.items()}
                        data = {k : json.loads(v) for k, v in message.items()}
                        logger.info(f"Response: {data}")

                        await process_message(message_id, data)
                        await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON for message {message_id}: {e}")

                    except Exception as e:
                        logger.error(f"Unexpected error for message {message_id}: {e}")
        except Exception as e:
            logger.exception(f"Error while reading new messages: {e}")


async def reclaim_stuck_messages(redis_client):
    while True:
        try:
            messages = await redis_client.xautoclaim(
                STREAM_NAME, GROUP_NAME, "consumer-1", min_idle_time=MIN_IDLE_TIME_MS,
                start_id="0-0", count=READ_COUNT
            )

            if not messages or not messages[1]:
                await asyncio.sleep(10)
                continue

            for message_id, message in messages[1]:
                logger.warning(f"Reclaiming and reprocessing stuck message: {message_id} , {message}")

                try:
                    # data = {k.decode(): v.decode() for k, v in message.items()}
                    data = {k : v for k, v in message.items()}
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON for message {message_id}")
                    continue

                await process_message(message_id, data)
                await redis_client.xack(STREAM_NAME, GROUP_NAME, message_id)

        except Exception as e:
            logger.exception(f"Error while reclaiming messages: {e}")
        
        await asyncio.sleep(5)


async def main():
    # redis_client = redis.from_url(REDIS_URI, decode_responses=True)
    redis_client = await get_redis_client()
    await ensure_group(redis_client)

    logger.info("Running with multiple consumers")
    consumer_names = ["consumer-1", "consumer-2"]

    # Start multiple consumers concurrently
    await asyncio.gather(
        *(consume_new_messages(redis_client, consumer_name) for consumer_name in consumer_names),
        reclaim_stuck_messages(redis_client)  # Reclaim stuck messages
    )


if __name__ == "__main__":
    asyncio.run(main())
