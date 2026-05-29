# train.py
import boto3, tarfile, os, pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Train
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 2. Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 3. Write inference.py that SageMaker calls on every prediction
with open("model/inference.py", "w") as f:
    f.write("""
import pickle, json, os

def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)

def predict_fn(input_data, model):
    return model.predict(input_data).tolist()

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    return data["instances"]

def output_fn(prediction, accept):
    return json.dumps(prediction), accept
""")

# 4. Package both files into tar.gz
with tarfile.open("trainedModel.tar.gz", "w:gz") as tar:
    tar.add("model/model.pkl", arcname="model.pkl")
    tar.add("model/inference.py", arcname="inference.py")

# 5. Upload to S3
s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]
s3.upload_file("trainedModel.tar.gz", bucket, "trainedModel.tar.gz")
print(f"Model uploaded to s3://{bucket}/trainedModel.tar.gz")