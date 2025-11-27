# Mock SageMaker Endpoint
#
# For AI/ML Scientists:
# This is a FAKE SageMaker endpoint that returns canned responses.
# It's NOT a real LLM - just simulates the API for local testing.
# Think of it as a "stub" or "mock" in unit testing.

FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir flask==3.0.0

# Copy mock server code
COPY server.py /app/
COPY responses/ /app/responses/

# Expose port
EXPOSE 8080

# Run mock server
CMD ["python", "server.py"]
