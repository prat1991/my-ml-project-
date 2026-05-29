# terraform/cd/main.tf
terraform {
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

variable "s3_bucket" {}
variable "sagemaker_role_arn" {}

# New unique suffix every CD run
resource "random_id" "deploy_id" {
  byte_length = 4
  keepers     = { always = timestamp() }
}

locals {
  suffix = random_id.deploy_id.hex
}

# 1. Model — unique name per run
resource "aws_sagemaker_model" "iris_model" {
  name               = "iris-model-${local.suffix}"
  execution_role_arn = var.sagemaker_role_arn

  primary_container {
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model_data_url = "s3://${var.s3_bucket}/trainedModel.tar.gz"
  }
}

# 2. Endpoint config — unique name per run
resource "aws_sagemaker_endpoint_configuration" "iris_config" {
  name = "iris-config-${local.suffix}"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.iris_model.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }
}

# 3. Endpoint — fixed name, rotates config each run
resource "aws_sagemaker_endpoint" "iris_endpoint" {
  name                 = "iris-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.iris_config.name

  lifecycle {
    create_before_destroy = true
  }
}

# 4. Run predict.py — pass endpoint name as argument
resource "null_resource" "predict" {
  depends_on = [aws_sagemaker_endpoint.iris_endpoint]
  triggers   = { always_run = timestamp() }

  provisioner "local-exec" {
    command = "pip install boto3 && python ../../predict.py ${aws_sagemaker_endpoint.iris_endpoint.name}"
  }
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.iris_endpoint.name
}