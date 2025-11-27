# Multi-Agent Lambda Container
#
# This Dockerfile packages the multi-agent workflow for AWS Lambda deployment.
#
# For AI/ML Scientists:
# Lambda containers let you package your code + dependencies into a Docker image.
# AWS runs this image when your function is invoked. Similar to deploying a
# model as a containerized service, but Lambda handles scaling automatically.

# Start from AWS-provided Python Lambda base image
# This image includes the Lambda runtime and Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# For AI/ML Scientists:
# Using AWS's base image ensures compatibility with Lambda's execution
# environment. It's like using an official PyTorch Docker image - saves you
# from configuration headaches.

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# For AI/ML Scientists:
# --no-cache-dir reduces image size by not storing pip's cache
# Important for Lambda: smaller images = faster cold starts

# Copy agent code from parent directory
# The multi-agent example uses sagemaker_llm.py from the main agent code
COPY ../../agent/sagemaker_llm.py .
COPY ../../agent/tools.py .

# Copy multi-agent specific code
COPY multi_agent_graph.py .
COPY agents/ agents/
COPY lambda_handler.py .

# For AI/ML Scientists:
# Lambda looks for code in ${LAMBDA_TASK_ROOT} (set by base image).
# We're copying:
# - Shared utilities (sagemaker_llm, tools) from main agent
# - Multi-agent graph definition
# - Individual agent modules
# - Lambda entry point

# Set the Lambda handler
CMD [ "lambda_handler.handler" ]

# For AI/ML Scientists:
# This tells Lambda which function to call when the container starts.
# Format: <filename>.<function_name>
# In our case: lambda_handler.py → handler() function

# -----------------------------------------------------------------------------
# Build and Deploy Instructions
# -----------------------------------------------------------------------------
#
# 1. Build the image locally:
#    docker build -t multi-agent-lambda .
#
# 2. Test locally (requires Docker Lambda runtime):
#    docker run -p 9000:8080 multi-agent-lambda
#    curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#      -d '{"body": "{\"question\": \"test\"}"}'
#
# 3. Tag for ECR:
#    docker tag multi-agent-lambda:latest \
#      <account>.dkr.ecr.<region>.amazonaws.com/multi-agent:latest
#
# 4. Push to ECR:
#    aws ecr get-login-password --region <region> | \
#      docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
#    docker push <account>.dkr.ecr.<region>.amazonaws.com/multi-agent:latest
#
# 5. Update Lambda function:
#    aws lambda update-function-code \
#      --function-name multi-agent-workflow \
#      --image-uri <account>.dkr.ecr.<region>.amazonaws.com/multi-agent:latest
#
# -----------------------------------------------------------------------------
# Size Optimization Tips
# -----------------------------------------------------------------------------
#
# If your image is too large (>10GB), Lambda won't accept it.
# Ways to reduce size:
#
# 1. Multi-stage build (advanced):
#    FROM python:3.11 AS builder
#    # Install dependencies here
#    FROM public.ecr.aws/lambda/python:3.11
#    COPY --from=builder /deps /deps
#
# 2. Remove unnecessary dependencies:
#    - Don't install dev dependencies (pytest, etc.)
#    - Use --no-deps for pip if you control all dependencies
#
# 3. Use slim versions of libraries where available
#
# Current image size: ~1.5GB (acceptable for Lambda)
