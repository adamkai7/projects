import boto3
import json
import os

S3_BUCKET = os.environ.get("S3_BUCKET", "my bucket") # hidden for public reasons
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "the link to the the function")# hidden for public reasons 
'''
this works with lambda_function.py to download earnings data
this file puts the 500 tickers in batches cause otherwise it will timeout

'''
def read_tickers_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        print(f"Reading tickers from s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        tickers = [ticker.strip().upper() for ticker in content.split('\n') if ticker.strip()]
        print(f"Found {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"Error reading tickers: {str(e)}")
        return []

def send_tickers_to_sqs(tickers, queue_url):
    sqs = boto3.client('sqs')
    
    # Send messages in batches of 10 (SQS limit)
    batch_size = 10
    total_sent = 0
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        entries = []
        
        for j, ticker in enumerate(batch):
            entries.append({
                'Id': str(i + j),
                'MessageBody': json.dumps({
                    'ticker': ticker,
                    'index': i + j,
                    'total_tickers': len(tickers)
                })
            })
        
        try:
            response = sqs.send_message_batch(
                QueueUrl=queue_url,
                Entries=entries
            )
            successful = len(response.get('Successful', []))
            total_sent += successful
            print(f"Sent batch {i//batch_size + 1}: {successful} messages")
            
        except Exception as e:
            print(f"Error sending batch {i//batch_size + 1}: {str(e)}")
    
    return total_sent

def lambda_handler(event, context):
    print("Starting ticker queue loader")
    
    # Read tickers from S3
    tickers = read_tickers_from_s3(S3_BUCKET, "ticker/sp500.txt")
    
    #tickers = read_tickers_from_s3(S3_BUCKET, "ticker/sp500.txt")[40:50]
    if not tickers:
        return {
            'statusCode': 400,
            'body': 'Failed to read tickers from S3'
        }
    
    # Send all tickers to SQS
    messages_sent = send_tickers_to_sqs(tickers, SQS_QUEUE_URL)
    
    print(f"Queue loading complete: {messages_sent} messages sent")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Successfully loaded tickers to queue',
            'total_tickers': len(tickers),
            'messages_sent': messages_sent,
            'queue_url': SQS_QUEUE_URL
        })
    }
