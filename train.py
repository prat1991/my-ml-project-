# train.py
import boto3, tarfile, os, pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Train
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 2. Save model file
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 3. Package as tar.gz (SageMaker requires this format)
with tarfile.open("trainedModel.tar.gz", "w:gz") as tar:
    tar.add("model/model.pkl", arcname="model.pkl")

# 4. Upload to S3
s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]          # set by Terraform output
s3.upload_file("trainedModel.tar.gz", bucket, "trainedModel.tar.gz")
print(f"Model uploaded to s3://{bucket}/trainedModel.tar.gz"){\rtf1}