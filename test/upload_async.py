import asyncio
import httpx
import uuid

async def upload_zip():
    url = 'http://localhost:8000/upload'

    # Form data
    form_data = {
        'job_id': '1235',
        'batch_name': 'python developer 2023',
    }

    # zip_path = r"C:\Users\Sourabh Kumar Das\Downloads\sample-local-pdf (2).zip"
    zip_path = "C:\\Users\\Sourabh Kumar Das\\Downloads\\sample-local-pdf (2).zip"

    # Prepare the file to send as multipart
    with open(zip_path, 'rb') as f:
        files = {
            'files': (f'{uuid.uuid4().hex[:5]}file.zip', f, 'application/zip')
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=form_data, files=files)

    print(f'Status code: {response.status_code}')
    print('Response body:', response.text)


async def upload_multiple(n: int):
    tasks = [upload_zip() for _ in range(n)]
    await asyncio.gather(*tasks)


async def upload_multiple_with_bg_task(n: int):
    print(n)
    # tasks = [asyncio.create_task(upload_zip()) for _ in range(n)]
    # await asyncio.gather(*tasks)

    tasks = []

    for i in range(n):
        print(f"Starting task {i}")
        await asyncio.sleep(0.1)
        asyncio.create_task(upload_zip())
        # task = asyncio.create_task(upload_zip())
        # tasks.append(task)

    # Wait for all background uploads to finish
    # await asyncio.gather(*tasks)

    await asyncio.sleep(10)



if __name__ == "__main__":
    # asyncio.run(upload_zip())
    # asyncio.run(upload_multiple(30))
    asyncio.run(upload_multiple_with_bg_task(20))
