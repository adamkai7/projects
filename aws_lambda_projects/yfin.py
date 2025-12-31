'''
Docstring for yfin


cp yfin.py lambda_function.py
zip yfin.zip lambda_function.py
aws lambda update-function-code --function-name yfin --zip-file fileb://yfin.zip
rm lambda_function.py

This runs it (but shouldnt ever need too)
aws lambda invoke --function-name yfin --cli-binary-format raw-in-base64-out --payload '{"ticker": "TSLA"}' response_yfin.json

Trigger/Batching: SQS sends a batch of messages. We create an empty list called batch_item_failures to 
keep track of only the specific messages that fail.

Processing: The Lambda iterates through these records. For each record, it extracts the ticker symbol.
the code acts like a web browser. It visits the Yahoo Earnings Calendar URL, and downloads the HTML.
It searches through the  HTML to find hidden JSON data inside <script> tags (where Yahoo hides the actual numbers) 
and extracts the earnings table.

Validation: It counts the rows. If there is only 1 row (just headers) or nothing, we say there is "No Data."

Success Path: If data is found, it converts it to CSV text and uploads it to the yf/earnings/ folder in S3.

Retry Logic 

Attempts < 3: If it fails, we add the Message ID to batch_item_failures. 
This tells SQS: "I failed this specific message. Please retry only this one after the Visibility Timeout (5 mins)."

Attempts >= 3: If it fails too many times, we create a .txt file in yf/errors/ with the error details.
 We do not add it to the failure list. This tells SQS: "We are done. Delete the message."


'''


import os
import json
import datetime
import requests
import boto3
import csv
import io
from bs4 import BeautifulSoup

# acting as browser stuff
S3_BUCKET = os.environ.get("S3_BUCKET", "atombucket123")
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def fetch_earnings(symbol):
    # Tries 3 internal times with short timeouts
    url = "https://finance.yahoo.com/calendar/earnings"
    params = {"symbol": symbol.replace(".", "-"), "size": 100}
    # Try fast (10s), then slower (20s), then slowest (30s)
    timeouts = [10, 20, 30]

    for i, timeout in enumerate(timeouts):
        try:
            print(f"Fetching {symbol} (Attempt {i+1} internal)...")
            # requests.get is the actual "Download" button
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()  # Crashes if we get a 404 or 500 error
            return response.text
        except requests.exceptions.RequestException:
            continue  # Try the next timeout

    raise TimeoutError(f"Failed to fetch {symbol} after internal retries")


def extract_earnings(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    # Yahoo hides data in <script> tags. Need to find all of them.
    script_tags = soup.find_all('script', attrs={'type': 'application/json'})

    for script in script_tags:
        # Loop through scripts until we find the one containing 'finance.yahoo.com/v1'
        data_url = script.attrs.get('data-url', '')
        if 'finance.yahoo.com/v1' not in data_url:
            continue

        try:
            outer_data = json.loads(script.text)
            body = outer_data.get('body', '')
            if not ('finance' in body and 'result' in body):
                continue
            inner_data = json.loads(body)
            if not inner_data['finance']['result']:
                continue

            doc_data = inner_data['finance']['result'][0]['documents'][0]
            columns = doc_data.get('columns', [])
            if not columns:
                continue

            headers = [col['id'] for col in columns]
            rows = doc_data.get('rows', [])
            # If  find the 'rows',  return them combined with 'headers'
            return [headers] + rows
        except:
            continue
    return None


def format_as_csv(data_list):
    # orignially a list of lists
    if not data_list:
        return None
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    for row in data_list:
        writer.writerow([str(cell) if cell is not None else '' for cell in row])
    return output.getvalue()


def upload_to_s3(content, bucket, key):
    # UPLOADS to s3 bucket
    boto3.client("s3").put_object(Body=content.encode('utf-8'), Bucket=bucket, Key=key, ContentType='text/csv')


def save_permanent_error(bucket, symbol, error_msg):
    # Saves a text file in S3 so you can see which specific ticker failed
    timestamp = datetime.datetime.now().strftime('%H%M%S')
    key = f"yf/errors/{symbol}_FAILURE_{timestamp}.txt"
    report = f"TICKER: {symbol}\nERROR: {error_msg}\nTIME: {datetime.datetime.utcnow().isoformat()}"
    boto3.client("s3").put_object(Body=report, Bucket=bucket, Key=key)


def lambda_handler(event, context):
    batch_item_failures = []
    # The list of messages want SQS to retry due to errors/failures

    # loops through everything in batch from SQS (right now batch size can varry from 1 to 10)
    for record in event['Records']:
        message_id = record['messageId']
       # this is th unique "Receipt Number" for that specific task.
       # Why it is important: It is the only way to tell SQS exactly which message to retry
        # In the new "Batch Failure" system you are using, the
        # set up allows this to process 10 messages at once. If 9 succeed and 1 fails,  need to tell AWS exactly
        # which one failed

        # This attribute tells us if this is the 1st, 2nd, or 3rd time SQS has sent this message
        # Check how many times SQS has sent this specific message
        attempts = int(record.get('attributes', {}).get('ApproximateReceiveCount', '1'))
        # 'ApproximateReceiveCount' tells us if is this the 1st time that this has been tried
        # or did we try this 5 minutes ago (current visability time out), fail, and now trying again?
        symbol = "UNKNOWN"
        # A safety default (or placeholder), if the message from SQS is completely corrupted
        #  The code tries to read body = json.loads(...) and crashes immediately.

        # It jumps to the except Exception as e: block.
        # The error block tries to print f"ERROR processing {symbol}...".

        # wouldnt work tbecayse thehe variable symbol was never created because the code crashed before it reached the line symbol = body.get('ticker').
        # Now  error logs are broken, and  no idea what happened

        try:
            # Parse the message body to get the ticker
            body = json.loads(record['body'])
            symbol = body.get('ticker')
            print(f"Processing: {symbol} (SQS Attempt {attempts})")
            # 1. Download HTML
            html = fetch_earnings(symbol)
            # 2. Extracts Data
            earnings_data = extract_earnings(html)

            # 3. Validate Data
            # If earnings_data is None OR it only contains 1 row (the headers), consider empty
            # This allows for fixes for broken tickers like AAAAA, as tickers that dont exsit will only have
            # no data or 1 item which is the head er in the csv
            if not earnings_data or len(earnings_data) <= 1:
                raise ValueError(f"Yahoo returned headers but NO data rows for {symbol}")

            # 4. Save to S3 and format as csv
            csv_content = format_as_csv(earnings_data)
            key = f"yf/earnings/{symbol}_{datetime.date.today()}.csv"
            upload_to_s3(csv_content, S3_BUCKET, key)
            print(f"SUCCESS: {symbol}")

        except Exception as e:
            print(f"ERROR processing {symbol}: {str(e)}")
            # CASE A:  Haven't failed enough times yet.
            # Attempt 1 failed. We want to try Attempt 2 also with longer time out just in case that was a problem
            # Logic: If we haven't hit the limit, report failure so SQS retries it later
            if attempts < MAX_RETRIES:
                # Add this specific message ID to the failure list
                # SQS will see this list, keep ONLY this message, wait 5 mins (Visibility Timeout),
                # and then send it to us again
                batch_item_failures.append({"itemIdentifier": message_id})
                # By adding this ID to the failure list, we tell SQS:
                # Keep this message, but hide it for 5 minutes (Visibility Timeout).
                # After 5 minutes, show it again so I can retry IT IS EFFECTIVLEY PUTTING BACK IN LINE

            # If we hit the limit, log to S3 and DO NOT report failure.
            # This tells SQS done with this message, delete it

            # CASE B: We have failed too many times.
            else:
                # We save a text file so know what happened
                save_permanent_error(S3_BUCKET, symbol, str(e))

                # Do NOT add this to 'batch_item_failures'
                # By NOT adding it, we tells SQS: We handled this
                # SQS will permanently delete the message so it stops looping
                print(f"GIVING UP on {symbol} after {attempts} attempts.")
                save_permanent_error(S3_BUCKET, symbol, str(e))

    return {"batchItemFailures": batch_item_failures}

    # hi
