# predict.py
import boto3, json, os

runtime = boto3.client("sagemaker-runtime")
endpoint = os.environ["SAGEMAKER_ENDPOINT"]   # set by Terraform output

# Sample iris measurement: sepal/petal length+width
payload = json.dumps({"instances": [[5.1, 3.5, 1.4, 0.2]]})

response = runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="application/json",
    Body=payload,
)
result = json.loads(response["Body"].read())
print("Prediction:", result)