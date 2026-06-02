import boto3, tarfile, os, pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# 1. Train
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 2. Save model
os.makedirs("model/code", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 3. Create inference.py — SageMaker calls these 4 functions automatically
with open("model/code/inference.py", "w") as f:
    f.write("""
import pickle, json, os
import numpy as np

def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)

def input_fn(request_body, content_type):
    return np.array(json.loads(request_body)["instances"])

def predict_fn(input_data, model):
    return model.predict(input_data).tolist()

def output_fn(prediction, accept):
    return json.dumps(prediction), accept
""")

# 4. Package — inference.py MUST be under code/ inside the tar
with tarfile.open("trainedModel.tar.gz", "w:gz") as tar:
    tar.add("model/model.pkl",        arcname="model.pkl")
    tar.add("model/code/inference.py", arcname="code/inference.py")

# 5. Upload to S3
s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]
s3.upload_file("trainedModel.tar.gz", bucket, "trainedModel.tar.gz")
print(f"Uploaded to s3://{bucket}/trainedModel.tar.gz")