# src/lambda_handler.py
import os
import json
import boto3
import tempfile
from datetime import datetime
from decimal import Decimal
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Use environment variables in Lambda configuration (safer)
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT")
DYNAMO_TABLE = os.environ.get("DYNAMODB_TABLE")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
REGION = os.environ.get("AWS_REGION", "us-east-1")

runtime = boto3.client("runtime.sagemaker", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)

def generate_serial_number(table):
    # a safe counter: use DynamoDB atomic counter or a timestamp-based id
    return str(int(datetime.utcnow().timestamp() * 1000))

def store_history(table_name, input_serialized, output_result):
    try:
        table = dynamodb.Table(table_name)
        item = {
            "ID": generate_serial_number(table_name),
            "Input": input_serialized,
            "Output": output_result,
            "Timestamp": datetime.utcnow().isoformat()
        }
        table.put_item(Item=item)
        logger.info("Saved record to DynamoDB: %s", item["ID"])
    except Exception as e:
        logger.exception("Failed to store to DynamoDB: %s", e)

def call_sagemaker(payload_csv):
    """
    payload_csv: single CSV line like "133.52,132.22,..." (string)
    """
    if not ENDPOINT_NAME:
        raise RuntimeError("SAGEMAKER_ENDPOINT env var not set")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload_csv
    )
    body = response["Body"].read().decode("utf-8")
    return body

def publish_notification(topic_arn, message, subject="Stock Prediction"):
    if not topic_arn:
        logger.warning("SNS topic ARN not configured, skipping publish")
        return
    try:
        sns.publish(TopicArn=topic_arn, Message=message, Subject=subject)
        logger.info("Published SNS notification")
    except Exception as e:
        logger.exception("Failed to publish SNS: %s", e)

def lambda_handler(event, context):
    """
    Expects event like:
    {
      "data": [
        [feature1, feature2, ...],
        ...
      ]
    }
    """
    try:
        body = event.get("data", event.get("body"))
        if isinstance(body, str):
            body = json.loads(body)

        inputs = body.get("data") if isinstance(body, dict) and "data" in body else body
        if not inputs:
            return {"statusCode": 400, "body": json.dumps({"error": "no input data"})}

        results = []
        for input_row in inputs:
            # convert row to CSV string expected by SageMaker endpoint
            csv_line = ",".join(map(str, input_row))
            logger.info("Calling SageMaker with: %s", csv_line)
            pred = call_sagemaker(csv_line)
            results.append(pred)

            # store history (best-effort)
            try:
                store_history(DYNAMO_TABLE, csv_line, pred)
            except Exception:
                logger.exception("history storage failed")

        # notify subscribers
        publish_notification(SNS_TOPIC_ARN, "Predictions: " + str(results), subject="Daily Stock Predictions")

        return {"statusCode": 200, "body": json.dumps({"predictions": results})}
    except Exception as e:
        logger.exception("Lambda failed: %s", e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
