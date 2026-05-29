# predict.py
import boto3, json, sys
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
endpoint = sys.argv[1]



payload = json.dumps({"instances": [[5.1, 3.5, 1.4, 0.2]]})

try:
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())
    print("Prediction:", result)
except Exception as e:
    print("Failed to call sagemaker prediction endpoint")