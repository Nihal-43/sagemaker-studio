# hander name -> lambda_function.lambda_handler
# python 3.11
# x86_64 Architecture
# Model: verseAI/databricks-dolly-v2-3b
# Go to Lambda configurations -> Environment variable -> replace ENDPOINT_NAME with sagemaker's endpoint name
# Import necessary libraries
import json
import boto3
import os
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a SageMaker client
sagemaker_client = boto3.client("sagemaker-runtime")

# Define Lambda function
def lambda_handler(event, context):
    # Log the incoming event in JSON format
    logger.info('Event: %s', json.dumps(event))
    
    # Parse the incoming request body
    try:
        request_body = json.loads(event['body'])
    except (json.JSONDecodeError, KeyError) as e:
        logger.error('Error parsing request body: %s', str(e))
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': 'Invalid request body'})
        }
    
    # Extract the prompt/instruction from the request
    # Support both "inputs" and "prompt" fields for flexibility
    prompt = request_body.get('inputs') or request_body.get('prompt', '')
    
    if not prompt:
        logger.error('No prompt provided in request')
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': 'No prompt provided'})
        }
    
    # Prepare payload for databricks-dolly-v2-3b model
    # The model expects: {"inputs": "prompt text", "parameters": {...}}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": request_body.get('max_new_tokens', 256),
            "temperature": request_body.get('temperature', 0.7),
            "top_p": request_body.get('top_p', 0.9),
            "do_sample": request_body.get('do_sample', True)
        }
    }
    
    # Log the payload being sent to SageMaker
    logger.info('Payload to SageMaker: %s', json.dumps(payload))

    # Invoke the SageMaker endpoint with the payload
    try:
        response = sagemaker_client.invoke_endpoint(
            EndpointName=os.environ["ENDPOINT_NAME"], 
            ContentType="application/json", 
            Body=json.dumps(payload)
        )
        
        # Load the response body and decode it
        result = json.loads(response["Body"].read().decode())
        
        logger.info('SageMaker response type: %s', type(result).__name__)
        logger.info('SageMaker response: %s', json.dumps(result))
        
        # databricks-dolly-v2-3b returns a plain string, wrap it in a structured format
        # Check if result is a string (plain text response) or already a dict
        if isinstance(result, str):
            # Try multiple common response formats that frontends expect
            formatted_result = [
                {
                    "generated_text": result
                }
            ]
        elif isinstance(result, list) and len(result) > 0:
            # Handle list response format [{"generated_text": "..."}]
            formatted_result = result if isinstance(result[0], dict) else [{"generated_text": result[0]}]
        else:
            # Already in dict format or other structure
            formatted_result = result
        
        logger.info('Formatted result type: %s', type(formatted_result).__name__)
        logger.info('Formatted result: %s', json.dumps(formatted_result))
        
        response_body = json.dumps(formatted_result)
        logger.info('Final response body: %s', response_body)
        
        # Return the result with status code 200 and the necessary headers
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': response_body
        }
    
    except Exception as e:
        logger.error('Error invoking SageMaker endpoint: %s', str(e))
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': 'Error invoking model endpoint'})
        }
