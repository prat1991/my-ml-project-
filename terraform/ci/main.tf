# terraform/ci/main.tf
terraform {
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    null   = { source = "hashicorp/null", version = "~> 3.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

# 1. Fixed bucket name — same every run, no random suffix
# CI and CD always agree on the bucket name
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "my-ml-models-iris"
  force_destroy = true
}

# 2. Train model and upload to S3
# numpy<2.0 required — SageMaker scikit-learn:1.2 container uses numpy 1.x
resource "null_resource" "train" {
  depends_on = [aws_s3_bucket.model_bucket]

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
	command = "pip install 'scikit-learn==1.2.*' boto3 'numpy<2.0' && python ..."
    environment = {
      S3_BUCKET = aws_s3_bucket.model_bucket.bucket
    }
  }
}

# 3. Output bucket name for CD
output "s3_bucket" {
  value = aws_s3_bucket.model_bucket.bucket
}
