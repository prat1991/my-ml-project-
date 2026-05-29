# terraform/ci/main.tf
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
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

  # THIS is the key fix — changes every run, so Terraform always re-executes
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

# 3. Output bucket name for cd/main.tf
output "s3_bucket" {
  value = aws_s3_bucket.model_bucket.bucket
}