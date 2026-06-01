# predict.py
import boto3, json, sys

# SageMaker runtime client — this is the interface to invoke deployed endpoints
# region must match where your endpoint is deployed
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Endpoint name passed in at runtime: python predict.py <endpoint-name>
# This keeps the code stateless — no hardcoded endpoint name
endpoint = sys.argv[1]

# The input payload — SageMaker expects {"instances": [...]} for sklearn/TF models
# Each inner list is one sample: [sepal_length, sepal_width, petal_length, petal_width]
# This is the Iris dataset format — 4 features per sample
request = json.dumps({"instances": [[5.1, 3.5, 1.4, 0.2]]})

try:
    # Send the feature vector to the deployed model on SageMaker
    # SageMaker runs inference and returns a prediction in the response Body
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,        # which deployed model to hit
        ContentType="application/json", # tells SageMaker how to parse the Body
        Body=request,                 # the actual feature vector
    )

    # Body is a streaming object — .read() pulls the raw bytes, json.loads parses it
    # result will be something like {"predictions": [0]} → class 0 = Iris Setosa
    result = json.loads(response["Body"].read())
    print("Prediction:", result)

except Exception as e:
    # Endpoint not found, wrong region, IAM permission missing, or malformed input
    print(f"Failed to call sagemaker endpoint locally or via cd pipeline")