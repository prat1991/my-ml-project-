# predict.py — invoke the endpoint AND monitor it (drift + accuracy)
# Original file just sent one sample and printed the prediction.
# Everything tagged "# NEW" below is the added monitoring requirement.
import boto3, json, sys, os
import numpy as np                      # NEW: for PSI / accuracy math
from sklearn.datasets import load_iris  # NEW: stand-in production batch

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")  # original
s3 = boto3.client("s3")                 # NEW: read baseline / write prod_data
endpoint = sys.argv[1]                  # original
bucket = os.environ["S3_BUCKET"]        # NEW: where baseline.json lives

# NEW: thresholds for the two rules
PSI_THRESHOLD = 0.2     # > 0.2 -> input distribution drifted -> retrain
ACC_THRESHOLD = 0.90    # < 90% -> predictions wrong -> labels need fixing

# NEW: a batch of recent production traffic as (features, true_label).
# In production this comes from SageMaker data capture + a labeling feedback
# loop. Here we sample 60 rows from Iris so the pipeline runs end to end.
# PSI needs a real batch (a handful of rows is just noise), hence 60.
Xall, yall = load_iris(return_X_y=True)
rng = np.random.default_rng(0)
idx = rng.choice(len(Xall), size=60, replace=True)
X = Xall[idx].astype(float)
y_true = yall[idx].tolist()
# To test the branches:
#   drift  -> X = X + 1.5          (PSI jumps past 0.2)
#   labels -> y_true = (yall[idx] ^ 1).tolist()   (corrupt labels -> accuracy drops)

# 1. Predictions from the live endpoint  (ORIGINAL invoke — now sends the batch)
resp = runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="application/json",
    Body=json.dumps({"instances": X.tolist()}),
)
y_pred = json.loads(resp["Body"].read())
print("Sample predictions:", y_pred[:10])

# ============================ NEW: monitoring ============================
# 2. DATA DRIFT — PSI of live features vs training baseline (saved by train.py)
s3.download_file(bucket, "baseline.json", "baseline.json")
baseline = json.load(open("baseline.json"))

psi = []
for c, b in enumerate(baseline):
    edges, exp = np.array(b["edges"]), np.array(b["expected"])
    cnt, _ = np.histogram(X[:, c], bins=edges)
    act = cnt / max(cnt.sum(), 1)
    e, a = np.clip(exp, 1e-6, None), np.clip(act, 1e-6, None)
    psi.append(float(np.sum((a - e) * np.log(a / e))))
max_psi = max(psi)
print(f"PSI per feature: {[round(p, 3) for p in psi]} (max={max_psi:.3f})")

# 3. ACCURACY DROP — predictions vs known true labels
acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
print(f"Accuracy: {acc:.3f}")

# 4. DECISION — accuracy first: if labels are wrong, more data won't help
if acc < ACC_THRESHOLD:
    decision = "fix_labels"          # stop -> human relabels training data
elif max_psi > PSI_THRESHOLD:
    decision = "retrain"             # retrain on combined train + production
    # production rows have no true label -> pseudo-label with model predictions
    combined = np.column_stack([X, np.array(y_pred)])
    np.savetxt("prod_data.csv", combined, delimiter=",")
    s3.upload_file("prod_data.csv", bucket, "prod_data.csv")
else:
    decision = "none"

print(f"DECISION={decision}")
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
    f.write(f"decision={decision}\n")
