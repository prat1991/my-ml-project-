# terraform/cd/main.tf
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

variable "s3_bucket" {
  description = "Bucket name output from CI"
}

variable "sagemaker_role_arn" {
  description = "IAM role ARN for SageMaker (create once in AWS console)"
}

# 1. Create SageMaker model (points to our .tar.gz in S3)
resource "aws_sagemaker_model" "iris_model" {
  name               = "iris-model"
  execution_role_arn = var.sagemaker_role_arn

  primary_container {
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    model_data_url = "s3://${var.s3_bucket}/trainedModel.tar.gz"
  }
}

# 2. Endpoint configuration (instance type + model)
resource "aws_sagemaker_endpoint_configuration" "iris_config" {
  name = "iris-endpoint-config"
  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.iris_model.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }
}

# 3. Deploy the endpoint
resource "aws_sagemaker_endpoint" "iris_endpoint" {
  name                 = "iris-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.iris_config.name
}

# 4. Run predict.py once endpoint is live
resource "null_resource" "predict" {
  depends_on = [aws_sagemaker_endpoint.iris_endpoint]

  provisioner "local-exec" {
    command = "pip install boto3 && python predict.py"
    environment = {
      SAGEMAKER_ENDPOINT = aws_sagemaker_endpoint.iris_endpoint.name
    }
  }
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.iris_endpoint.name
}