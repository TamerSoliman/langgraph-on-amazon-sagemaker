# Dockerfile for LangGraph Agent (AWS Lambda Container)
#
# This container runs the LangGraph agent code on AWS Lambda.
# It's optimized for size and cold start time.
#
# For AI/ML Scientists:
# - This is NOT the LLM container (that's on SageMaker)
# - This container only has Python code for orchestration (LangGraph logic)
# - No GPU drivers, no PyTorch, no model weights
# - Keeps it lightweight (~500MB) for fast Lambda deployments

# Use AWS Lambda Python base image
# This image is optimized for Lambda and includes the Lambda Runtime Interface Client
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first (Docker layer caching - only rebuilds if requirements change)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't store pip cache (reduces image size)
# --upgrade: Get latest compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY graph.py ${LAMBDA_TASK_ROOT}/
COPY tools.py ${LAMBDA_TASK_ROOT}/
COPY sagemaker_llm.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
# AWS Lambda will call: lambda_handler.handler(event, context)
CMD [ "lambda_handler.handler" ]
