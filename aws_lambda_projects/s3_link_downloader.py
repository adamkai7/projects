import os
import urllib.request
import boto3
import datetime
from io import BytesIO

s3 = boto3.client('s3')

BUCKET_NAME = "bucket name" # hidden for security reasons 
LINKS_FILE_KEY = "links.txt"
FAILED_LINKS_KEY = "failed_links.txt"

def lambda_handler(event, context):
    print("Starting Lambda to download links from links.txt")

    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=LINKS_FILE_KEY)
        file_content = response['Body'].read().decode('utf-8')
        urls = [line.strip() for line in file_content.splitlines() if line.strip()]
        print(f"Found {len(urls)} URLs")
    except Exception as e:
        print(f"Error reading links.txt: {e}")
        return {"statusCode": 500, "body": f"Error reading links.txt: {str(e)}"}

    failed_links = []
    results = []
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for url in urls:
        filename = url.split('/')[-1] or f"file_{timestamp}"
        tmp_path = f"/tmp/{filename}"

        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, tmp_path)
        except Exception as e:
            print(f"First download attempt failed for {url}: {e}")
            try:
                print(f"Retrying download for {url}")
                urllib.request.urlretrieve(url, tmp_path)
            except Exception as e2:
                print(f"Retry failed for {url}: {e2}")
                failed_links.append(url)
                results.append({"url": url, "status": "error", "error": str(e2)})
                continue

        try:
            # Save with timestamp to avoid overwriting
            s3_key = f"downloads/{timestamp}_{filename}"
            s3.upload_file(tmp_path, BUCKET_NAME, s3_key)
            print(f"Uploaded to s3://{BUCKET_NAME}/{s3_key}")
            results.append({"url": url, "status": "success", "s3_key": s3_key})
        except Exception as e:
            print(f"Error uploading {url} to S3: {e}")
            failed_links.append(url)
            results.append({"url": url, "status": "upload_error", "error": str(e)})

    if failed_links:
        existing_data = ""
        try:
            existing_file = s3.get_object(Bucket=BUCKET_NAME, Key=FAILED_LINKS_KEY)
            existing_data = existing_file['Body'].read().decode('utf-8')
        except s3.exceptions.NoSuchKey:
            pass

        new_data = existing_data + "\n".join(failed_links) + "\n"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=FAILED_LINKS_KEY,
            Body=new_data.encode('utf-8')
        )
        print(f"Logged {len(failed_links)} failed links to {FAILED_LINKS_KEY}")

    print(f"Finished downloading {len(results)} files.")
    return {
        "statusCode": 200,
        "body": f"Downloaded {len(results)} files to s3://{BUCKET_NAME}/downloads/, failed: {len(failed_links)}"
    }

