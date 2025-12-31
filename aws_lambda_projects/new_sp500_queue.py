import boto3
import json
import os
import time
from datetime import datetime
'''
Builds a temporary SQS queue just for this specific run
Connects this new queue to Worker Function (yfin)
Reads the list of stocks from S3 and pushes them into the queue
Waits 30 seconds to allow AWS internals to sync up (solving the "Race Condition")
Monitor the queue until it is completely empty (0 pending, 0 running).
Disconnects the worker and deletes the queue
'''
import boto3
import json
import os
import time
from datetime import datetime

# --- CONFIG ---
S3_BUCKET = os.environ.get("S3_BUCKET", "atombucket123")
WORKER_FUNCTION = "yfin"
TICKER_FILE = "ticker/sp500.txt"


def lambda_handler(event, context):

    # c reate/delete queues
    sqs = boto3.client("sqs")
    # worker function
    lambda_client = boto3.client("lambda")
    # read text file
    s3 = boto3.client("s3")

    # Create a temporary queue with a time stamp so the name is unique
    timestamp = datetime.now().strftime("%H%M%S")
    queue_name = f"temp-queue-{timestamp}"

    print(f"Creating Queue: {queue_name}")

    # IT MUST BE GREATER THAN OR EQUAL TO THE WORKER FUNCTION TIME OUT
    q_resp = sqs.create_queue(QueueName=queue_name, Attributes={'VisibilityTimeout': '900'})
    queue_url = q_resp['QueueUrl']

    # have to get ARN which allows for permssions and linking, the url sends the msgs
    q_attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['QueueArn'])
    queue_arn = q_attrs['Attributes']['QueueArn']

    mapping_uuid = None

    try:
        # This will trigger the worker Lambda from this new queue

        print(f"Connecting '{WORKER_FUNCTION}'...")
        mapping = lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=WORKER_FUNCTION,
            Enabled=True,
            BatchSize=10
        )
        mapping_uuid = mapping['UUID']
        # This is the ID so can delete the link later

        time.sleep(5)  # Allow connection to propagate

        # Read S3 and push to SQS
        print("Sending tickers")
        # This just downloads and and cleans the text file (remove spaces/newlines)
        file_content = s3.get_object(Bucket=S3_BUCKET, Key=TICKER_FILE)['Body'].read().decode('utf-8')
        tickers = [t.strip().upper() for t in file_content.split('\n') if t.strip()]

        # SQS Batching Loop:

        for i in range(0, len(tickers), 10):
            batch = tickers[i:i+10]
            # Format into the JSON structure SQS requires
            entries = [{'Id': str(i+j), 'MessageBody': json.dumps({'ticker': t})} for j, t in enumerate(batch)]
            sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        print(f"   Sent {len(tickers)} tickers.")

        # SQS is "Eventually Consistent." When you dump 500 items, the "Count" metric
        # might still read "0" for 10-20 seconds while AWS servers sync up.
        # We wait 30 seconds blindly to ensure the counter catches up to reality.
        print("4. Allowing SQS 30s to update metrics...")
        time.sleep(30)

        # 4. WAIT: Loop until done
        print("5. Monitoring for completion...")
        while True:
            if context.get_remaining_time_in_millis() < 10000:
                print("WARNING: Lambda timeout imminent! Stopping wait to clean up.")
                break

            status = sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )
            pending = int(status['Attributes']['ApproximateNumberOfMessages'])
            running = int(status['Attributes']['ApproximateNumberOfMessagesNotVisible'])

            print(f"   Status: {pending} waiting, {running} running...")

            # Only quit if BOTH are zero
            if pending == 0 and running == 0:
                # Double Check: Wait 5s and check one last time to be sure it wasn't a glitch
                time.sleep(5)
                status_check = sqs.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
                )
                p2 = int(status_check['Attributes']['ApproximateNumberOfMessages'])
                r2 = int(status_check['Attributes']['ApproximateNumberOfMessagesNotVisible'])

                if p2 == 0 and r2 == 0:
                    print("   Job Complete (Confirmed)!")
                    break
                else:
                    print("   False Alarm! Work appeared. Resuming wait...")

            # Wait 10s between checks to save logs
            time.sleep(10)

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        pass

    finally:
        # 5. CLEANUP
        print("6. Cleaning up...")
        if mapping_uuid:
            try:
                lambda_client.delete_event_source_mapping(UUID=mapping_uuid)
                print("   Disconnected Lambda.")
            except Exception as e:
                print(f"   Failed to disconnect: {e}")

        try:
            sqs.delete_queue(QueueUrl=queue_url)
            print("   Deleted Queue.")
        except Exception as e:
            print(f"   Failed to delete queue: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps('Orchestration finished')
    }
