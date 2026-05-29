# terraform/ci/main.tf
terraform {
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.0" }
    null   = { source = "hashicorp/null", version = "~> 3.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

# 1. Create S3 bucket (created once, reused across runs)
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "my-ml-models-${random_id.suffix.hex}"
  force_destroy = true
}

resource "random_id" "suffix" {
  byte_length = 4
}

# 2. Force retrain on every CI run by using a timestamp trigger
resource "null_resource" "train" {
  depends_on = [aws_s3_bucket.model_bucket]

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = "pip install scikit-learn boto3 && python ${path.root}/../../train.py"
    environment = {
      S3_BUCKET = aws_s3_bucket.model_bucket.bucket
    }
  }
}

# 3. Repack model correctly after training
resource "null_resource" "repack" {
  depends_on = [null_resource.train]

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<EOT
      set -e

      # 1. Create correct folder structure
      mkdir -p model_package/code

      # 2. Copy model and inference script
      cp model.pkl model_package/
      cp ${path.root}/../../inference.py model_package/code/

      # 3. Verify inference.py is there
      ls -la model_package/code/

      # 4. Repack correctly — no extra wrapper layer
      cd model_package
      tar -czvf ../trainedModel.tar.gz code/ model.pkl

      # 5. Verify structure
      cd ..
      echo "--- Verifying tar.gz structure ---"
      tar -tzf trainedModel.tar.gz

      # 6. Upload to S3
      aws s3 cp trainedModel.tar.gz s3://${aws_s3_bucket.model_bucket.bucket}/trainedModel.tar.gz

      echo "--- Upload complete ---"
    EOT
  }
}

# 4. Output bucket name for cd/main.tf
output "s3_bucket" {
  value = aws_s3_bucket.model_bucket.bucket
}
