import boto3, tarfile, os, pickle, json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]

# 1. Train data
X, y = load_iris(return_X_y=True)

# 1b. NEW: if a previous monitor run found drift, it left pseudo-labeled
#     production rows in S3 — fold them in, then delete so we don't reuse them.
try:
    s3.download_file(bucket, "prod_data.csv", "prod_data.csv")
    prod = np.loadtxt("prod_data.csv", delimiter=",")
    X = np.vstack([X, prod[:, :-1]])
    y = np.concatenate([y, prod[:, -1].astype(int)])
    s3.delete_object(Bucket=bucket, Key="prod_data.csv")
    print(f"Retraining on COMBINED data: {len(X)} rows")
except s3.exceptions.ClientError:
    print(f"Training on base data: {len(X)} rows")

model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X, y)

# 2. Save model
os.makedirs("model/code", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 2b. NEW: save the training input distribution (bin edges + % per bin per
#     feature). predict.py reads this to compute PSI against live traffic.
baseline = []
for c in range(X.shape[1]):
    edges = np.unique(np.quantile(X[:, c], np.linspace(0, 1, 11)))
    cnt, _ = np.histogram(X[:, c], bins=edges)
    baseline.append({"edges": edges.tolist(), "expected": (cnt / cnt.sum()).tolist()})
with open("baseline.json", "w") as f:
    json.dump(baseline, f)

# 3. inference.py — SageMaker calls these 4 functions automatically (unchanged)
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

# 5. Upload model + baseline
s3.upload_file("trainedModel.tar.gz", bucket, "trainedModel.tar.gz")
s3.upload_file("baseline.json", bucket, "baseline.json")   # NEW
print(f"Uploaded to s3://{bucket}/trainedModel.tar.gz")
