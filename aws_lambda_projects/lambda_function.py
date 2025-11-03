import os
import json
import datetime
import requests
import boto3
import csv
import io
from bs4 import BeautifulSoup


S3_BUCKET = os.environ.get("S3_BUCKET", "MY BUCKET NAME")# for the public/github version im hiding the name of my bucket

'''
this code downloads and works with another file (sp500-queue-loader) to download earnings data from all tickers in the sp500
it will probably break at some point because yahoo finance changes their stuff all the time
'''
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_earnings(symbol, size=100):
    #reads the tickers and fetch earnings
    url = "https://finance.yahoo.com/calendar/earnings"
    params = {
        "symbol": symbol.replace(".", "-"),
        "size": size
    }
    response = requests.get(url, params=params, headers=HEADERS, timeout=10)  
    response.raise_for_status()
    return response.text


def extract_earnings(html_text):
    # the earnings are saved as a html and table, need to parse it

    soup = BeautifulSoup(html_text, 'html.parser')
    script_tags = soup.find_all('script', attrs={'type': 'application/json'})

    for script in script_tags:
        data_url = script.attrs.get('data-url', '')
        if 'finance.yahoo.com/v1' not in data_url:
            continue

        try:
            outer_data = json.loads(script.text)
            body = outer_data.get('body', '')

            if not ('finance' in body and 'result' in body):
                continue

            inner_data = json.loads(body)
            nearning = inner_data['finance']['result'][0]['total']
            doc_data = inner_data['finance']['result'][0]['documents'][0]

        except:
            continue

        entity_id_type = doc_data.get('entityIdType', '')
        if entity_id_type != 'SP_EARNINGS' or nearning == 0:
            continue

        columns = doc_data.get('columns', [])
        if not columns:
            continue

        headers = [col['id'] for col in columns]
        rows = doc_data.get('rows', [])
        if not rows:
            return None

        data_list = [headers] + rows
        return data_list

    return None


def format_as_csv(data_list):
    # saves
    if not data_list:
        return None

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    for row in data_list:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append('')
            else:
                cleaned_row.append(str(cell))
        writer.writerow(cleaned_row)

    csv_content = output.getvalue()
    output.close()
    return csv_content


def upload_to_s3(csv_content, bucket, key):
    #uploads to my bit bucket
    s3 = boto3.client("s3")
    s3.put_object(
        Body=csv_content.encode('utf-8'),
        Bucket=bucket,
        Key=key,
        ContentType='text/csv'
    )
    print(f"Uploaded to s3://{bucket}/{key}")


def lambda_handler(event, context):
    """
    Process SQS messages containing individual ticker symbols
    """
    print("Lambda function started")
    print(f"Received event: {json.dumps(event)}")
    print("hello world")
    # Process SQS records
    if 'Records' not in event:
        print("No SQS records found in event")
        return {'statusCode': 400, 'body': 'No SQS records found'}
    
    # Initialize SQS client for message deletion
    sqs = boto3.client('sqs')
    queue_url = os.environ.get('SQS_QUEUE_URL', 'https://sqs.us-east-2.amazonaws.com/590183791055/sp500-ticker-queue')
    
    results = []
    
    for record in event['Records']:
        try:
            # Parse the SQS message
            message_body = json.loads(record['body'])
            symbol = message_body.get('ticker')
            
            if not symbol:
                print(f"No ticker found in message: {message_body}")
                continue
                
            print(f"Processing symbol: {symbol}")
            
            # Process the ticker
            key = f"yf/earnings/{symbol}_{datetime.datetime.utcnow().isoformat()}.csv"
            html_content = fetch_earnings(symbol)
            earnings_data = extract_earnings(html_content)

            if not earnings_data:
                results.append({"status": "no_data", "symbol": symbol})
                print(f"No earnings data found for {symbol}")
                continue

            csv_content = format_as_csv(earnings_data)
            upload_to_s3(csv_content, S3_BUCKET, key)

            results.append({
                "status": "success",
                "symbol": symbol,
                "s3_key": key,
                "records": len(earnings_data) - 1
            })
            print(f"Successfully processed {symbol}")
            
        except Exception as e:
            print(f"Error processing record {record}: {str(e)}")
            results.append({
                "status": "error", 
                "symbol": record.get('body', 'unknown'),
                "error": str(e)
            })
            # Don't delete message on error - let it retry
            raise e

    print(f"Processed {len(results)} symbols from SQS")
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

