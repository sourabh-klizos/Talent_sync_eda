import asyncio
import httpx
import uuid


error_count, success_count = 0, 0

_success_lock = asyncio.Lock()
_error_lock = asyncio.Lock()


async def upload_zip():
    try:

        url = "http://localhost:8000/upload"

        # Form data
        form_data = {
            "job_id": "1235",
            "batch_name": "python developer 2023",
        }

        # zip_path = r"C:\Users\Sourabh Kumar Das\Downloads\sample-local-pdf (2).zip"
        # zip_path = "C:\\Users\\Sourabh Kumar Das\\Downloads\\sample-local-pdf (2).zip"
        zip_path = r"C:\Users\Sourabh Kumar Das\Downloads\sample-report_new.zip"

        # Prepare the file to send as multipart
        with open(zip_path, "rb") as f:
            files = {"files": (f"{uuid.uuid4().hex[:5]}file.zip", f, "application/zip")}

            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=form_data, files=files)

        if response.status_code and response.status_code == 200:
            # with _success_lock:
            global success_count
            success_count += 1

            # print(f"Status code: {response.status_code}")
            # print("Response body:", response.text)
        else:
            # with _error_lock:
            global error_count
            error_count += 1
            # print("Error Response :", response.text)
            # print(f"Status code: {response.status_code}")

        return response

    except Exception as e:
        pass
        # print("Error Response :", response.text)
        # print(f"Status code: {response.status_code}")


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
        # await asyncio.sleep(0.2)
        # asyncio.create_task(upload_zip())

        tasks.append(asyncio.create_task(upload_zip()))

    results = await asyncio.gather(*tasks)
    # print("done- > ")

    for result in results:
        if result:
            print(f"Status code: {result.status_code}")
            print("Res body:", result.text)

    # await asyncio.sleep(100)


if __name__ == "__main__":
    # asyncio.run(upload_zip())
    # asyncio.run(upload_multiple(30))

    asyncio.run(upload_multiple_with_bg_task(20))

    print(f"error_count: -> {error_count}")

    print(f"success_count: -> {success_count}")
