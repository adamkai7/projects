'''
cd to where this file is stored:

cp sp500_queue_loader.py lambda_function.py
zip sp500.zip lambda_function.py
aws lambda update-function-code --function-name sp500-queue-loader --zip-file fileb://sp500.zip
rm lambda_function.py


This runs it:
aws lambda invoke --function-name sp500-queue-loader response_sp500.json

how to view files:

aws s3 ls s3://atombucket123/yf/earnings/
'''

import boto3
import json
import os

# where the bucket is

S3_BUCKET = os.environ.get("S3_BUCKET", "atombucket123")
# Adress of the queue, sends the batches to here

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/590183791055/sp500-ticker-queue")


def read_tickers_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        print(f"Reading tickers from s3://{bucket}/{key}")
        # 1. Download
        # get_object stream-reads the file from S3.
        response = s3.get_object(Bucket=bucket, Key=key)
        # 2. Decode
        # S3 returns bytes, need to decode it to UTF-8 text
        content = response['Body'].read().decode('utf-8')
        # 3. Clean & split
        # Split the big text block by newlinesto get individual lines
        # .strip() removes invisible spaces and .upper() ensures 'tsla' becomes 'TSLA' so its a ticker that
        # will work in Yahoo
        tickers = [ticker.strip().upper() for ticker in content.split('\n') if ticker.strip()]
        print(f"Found {len(tickers)} tickers")
        return tickers
    except Exception as e:
        # If the file is missing or S3 is down,  return an empty list to prevent a crash or problem
        print(f"Error reading tickers: {str(e)}")
        return []


def send_tickers_to_sqs(tickers, queue_url):
    # SQS Limit: You can only send 10 messages per API call
    sqs = boto3.client('sqs')
    batch_size = 10
    total_sent = 0

    for i in range(0, len(tickers), batch_size):
        # Steps through the list 10 items at a time (0, 10, 20, 30
        # Slice the list to get just the next 10 items
        batch = tickers[i:i + batch_size]
        entries = []

        for j, ticker in enumerate(batch):
            # Format each of the 10 items for SQS

            entries.append({
                'Id': str(i + j),
                'MessageBody': json.dumps({
                    # 'MessageBody': The actual data the downloader (yfin) will receive
                    # Dump it to a JSON string because SQS only accepts text
                    'ticker': ticker,
                    'index': i + j,
                    'total': len(tickers)
                })
            })

        try:  # Send: The actual API call that puts 10 messages in the queue
            sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
            total_sent += len(entries)
            print(f"Sent batch {i//batch_size + 1} ({len(entries)} tickers)")
        except Exception as e:
            print(f"Error sending batch {i//batch_size + 1}: {str(e)}")

    return total_sent


def lambda_handler(event, context):
    #  looks for a file named "sp500.txt" inside a folder named "ticker"
    print("Starting ticker queue loader")
    tickers = read_tickers_from_s3(S3_BUCKET, "ticker/sp500.txt")
# Safety: If S3 failed or file was empty, stop here
    if not tickers:
        return {'statusCode': 400, 'body': 'Failed to read tickers'}
    # dispatch
    messages_sent = send_tickers_to_sqs(tickers, SQS_QUEUE_URL)
# report sucess
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Successfully loaded queue',
            'total_tickers': len(tickers),
            'sent': messages_sent
        })
    }
