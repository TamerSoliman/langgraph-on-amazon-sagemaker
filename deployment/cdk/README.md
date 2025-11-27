# AWS CDK Deployment for LangGraph on SageMaker

## Overview

This directory contains AWS CDK (Cloud Development Kit) code to deploy the complete LangGraph on SageMaker stack with **one command**.

**For AI/ML Scientists:** CDK is "infrastructure as code" - you write TypeScript to define AWS resources instead of clicking through the AWS console. Think of it as a Dockerfile for cloud infrastructure.

---

## What Gets Deployed

Running `cdk deploy --all` creates:

1. **Secrets Stack** - Stores API keys (Tavily) in AWS Secrets Manager
2. **SageMaker Endpoint** - Deploys Mistral 7B LLM on GPU (ml.g5.xlarge)
3. **Lambda Function** - Runs your LangGraph agent code (serverless)
4. **API Gateway** - Public HTTPS API for asking questions
5. **Monitoring** - CloudWatch dashboards + alarms + email notifications

**Total monthly cost:** ~$730 (SageMaker endpoint is 96% of this)

---

## Prerequisites

### 1. Install Node.js and AWS CDK

```bash
# Install Node.js (version 18 or higher)
# macOS:
brew install node

# Linux:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version  # Should be v18.x or higher
npm --version

# Install AWS CDK globally
npm install -g aws-cdk

# Verify CDK installation
cdk --version  # Should be 2.100.0 or higher
```

### 2. Configure AWS Credentials

```bash
# Install AWS CLI if not already installed
# macOS:
brew install awscli

# Linux:
sudo apt-get install awscli

# Configure credentials (you'll need AWS Access Key ID and Secret)
aws configure
# Enter your AWS Access Key ID
# Enter your Secret Access Key
# Default region: us-east-1
# Default output format: json

# Verify credentials work
aws sts get-caller-identity
# Should show your AWS account ID
```

### 3. Bootstrap CDK (One-Time Setup)

```bash
# This creates an S3 bucket and other resources CDK needs
# Only run this once per AWS account + region
cdk bootstrap aws://YOUR-ACCOUNT-ID/us-east-1

# If you don't know your account ID:
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
cdk bootstrap aws://$AWS_ACCOUNT/us-east-1
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd deployment/cdk
npm install
```

This installs all TypeScript dependencies listed in `package.json`.

### 2. Build TypeScript Code

```bash
npm run build
```

This compiles TypeScript (`.ts` files) to JavaScript (`.js` files).

### 3. Review What Will Be Created (Optional)

```bash
cdk synth
```

This generates CloudFormation templates showing exactly what will be deployed. The output is saved to `cdk.out/`.

### 4. Preview Changes

```bash
cdk diff
```

This shows what will change if you deploy (useful when updating existing stacks).

### 5. Deploy Everything

```bash
# Set your email for alarm notifications
export ALARM_EMAIL="your-email@example.com"

# Deploy all stacks
npm run deploy

# OR manually:
cdk deploy --all --require-approval never
```

**What happens:**
1. CDK builds your Lambda Docker image (~5 minutes)
2. Pushes image to Amazon ECR (Elastic Container Registry)
3. Creates CloudFormation stacks (~10-15 minutes total)
4. SageMaker endpoint takes the longest (5-10 minutes)

**Expected output:**
```
✅  langgraph-dev-secrets
✅  langgraph-dev-sagemaker
✅  langgraph-dev-lambda
✅  langgraph-dev-api
✅  langgraph-dev-monitoring

Outputs:
langgraph-dev-api.ApiUrl = https://abc123.execute-api.us-east-1.amazonaws.com/prod/
langgraph-dev-api.CurlExample = curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/prod/ask ...
langgraph-dev-sagemaker.EndpointName = langgraph-huggingface-llm-mistral-7b-instruct-endpoint
...
```

### 6. Set the Tavily API Key

The deployment creates a secret in AWS Secrets Manager, but you need to set the value:

```bash
# Get your Tavily API key from https://app.tavily.com/

# Set the secret value
aws secretsmanager put-secret-value \
  --secret-id langgraph/tavily-api-key \
  --secret-string '{"api_key":"tvly-YOUR_ACTUAL_KEY_HERE"}'
```

### 7. Test the Deployment

```bash
# Get the API URL from the deployment output
API_URL="<your-api-url-from-step-5>"

# Test with curl
curl -X POST ${API_URL}ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'

# Expected response:
# {"answer":"The capital of France is Paris.","metadata":{...}}
```

---

## Project Structure

```
deployment/cdk/
├── bin/
│   └── app.ts                  # Main entry point (orchestrates all stacks)
├── lib/
│   ├── secrets-stack.ts        # Creates Secrets Manager secrets
│   ├── sagemaker-endpoint-stack.ts  # Deploys Mistral 7B model
│   ├── lambda-agent-stack.ts   # Deploys LangGraph agent as Lambda
│   ├── api-gateway-stack.ts    # Creates REST API
│   └── monitoring-stack.ts     # Sets up dashboards + alarms
├── package.json                # Node.js dependencies
├── tsconfig.json               # TypeScript compiler config
├── cdk.json                    # CDK configuration
└── README.md                   # This file
```

---

## Environment-Specific Deployments

Deploy to different environments (dev, staging, prod) with different configurations:

```bash
# Development (smaller instance, shorter log retention)
cdk deploy --all --context environment=dev

# Production (larger instance, longer log retention, stricter alarms)
cdk deploy --all --context environment=prod

# Custom instance type
cdk deploy --all --context environment=dev --context instanceType=ml.g5.2xlarge
```

Environment affects:
- Stack naming: `langgraph-dev-*` vs `langgraph-prod-*`
- Instance type: ml.g5.xlarge (dev) vs ml.g5.2xlarge (prod)
- Log retention: 7 days (dev) vs 30 days (prod)
- Alarm thresholds: Relaxed (dev) vs Strict (prod)

---

## Cost Management

### Estimated Monthly Costs (us-east-1)

| Component | Cost/Month | Usage Pattern |
|-----------|------------|---------------|
| **SageMaker Endpoint** (ml.g5.xlarge) | $723 | 24/7 operation |
| **Lambda** | $0.35 | 1,000 questions/day |
| **API Gateway** | $0.10 | 30,000 requests/month |
| **CloudWatch** | $3.00 | Logs + dashboards + alarms |
| **Secrets Manager** | $0.40 | 1 secret |
| **ECR** (Docker images) | $0.10 | ~500MB |
| **Total** | **~$727/month** | |

### Cost Optimization Tips

1. **Use Serverless Endpoints** (Preview feature - only pay for inference time):
   ```typescript
   // In sagemaker-endpoint-stack.ts
   // Replace real-time endpoint with serverless endpoint
   // Saves ~$700/month but has cold start (5-10 seconds)
   ```

2. **Stop endpoint when not in use:**
   ```bash
   # Stop endpoint (no charges while stopped)
   aws sagemaker delete-endpoint --endpoint-name <endpoint-name>

   # Redeploy when needed
   cdk deploy langgraph-dev-sagemaker
   ```

3. **Use smaller instance for dev/testing:**
   ```bash
   # ml.g5.xlarge = $723/month
   # Consider using JumpStart UI during development
   # Then deploy with CDK for production only
   ```

4. **Set up budget alerts:**
   ```bash
   # Create a budget in AWS Budgets
   aws budgets create-budget \
     --account-id $(aws sts get-caller-identity --query Account --output text) \
     --budget file://budget.json
   ```

---

## Monitoring

### View CloudWatch Dashboard

```bash
# Get dashboard URL from deployment output
# OR construct it:
REGION="us-east-1"
DASHBOARD_NAME="langgraph-agent-dashboard"

echo "https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=${DASHBOARD_NAME}"
```

Dashboard shows:
- Request count (API Gateway + Lambda + SageMaker)
- Error rates (4xx, 5xx)
- Latency (API, Lambda, SageMaker)
- Cost trends

### View Logs

```bash
# Lambda logs
aws logs tail /aws/lambda/langgraph-agent --follow

# API Gateway access logs
aws logs tail /aws/apigateway/langgraph-agent --follow

# Filter for errors only
aws logs tail /aws/lambda/langgraph-agent --follow --filter-pattern "ERROR"
```

### Alarm Notifications

When an alarm fires, you'll receive an email at the address you specified. The email includes:
- Which alarm fired (e.g., "LambdaErrorAlarm")
- Current metric value (e.g., "10% error rate")
- Threshold (e.g., "5%")
- Link to CloudWatch dashboard

**IMPORTANT:** Check your email after deployment and **confirm the SNS subscription** to start receiving alerts!

---

## Updating the Stack

### Update Agent Code

After changing your agent code (in `agent/` directory):

```bash
# Rebuild and redeploy Lambda only
cdk deploy langgraph-dev-lambda
```

CDK automatically:
1. Rebuilds the Docker image
2. Pushes new image to ECR
3. Updates Lambda to use new image
4. No downtime (Lambda auto-switches)

### Update Infrastructure (CDK Code)

After changing CDK code (in `lib/` directory):

```bash
# Rebuild TypeScript
npm run build

# Preview changes
cdk diff

# Deploy changes
cdk deploy --all
```

---

## Troubleshooting

### Issue: "Error: Need to perform AWS calls for account X, but no credentials configured"

**Solution:** Run `aws configure` and enter your credentials.

---

### Issue: "Error: This stack uses assets, so the toolkit stack must be deployed"

**Solution:** Run `cdk bootstrap aws://YOUR-ACCOUNT-ID/us-east-1`

---

### Issue: "Lambda deployment failed: RequestEntityTooLargeException"

**Cause:** Docker image > 10GB

**Solution:** Reduce dependencies in `agent/requirements.txt`. Remove unused packages.

---

### Issue: "SageMaker endpoint stuck in 'Creating' state"

**Cause:** Model download/initialization can take 5-10 minutes

**Solution:** Wait. Check CloudWatch logs:
```bash
aws logs tail /aws/sagemaker/Endpoints/langgraph-huggingface-llm-mistral-7b-instruct-endpoint --follow
```

If it fails, check:
- Instance type has enough VRAM (Mistral 7B needs ~24GB)
- IAM role has permissions to download model from S3

---

### Issue: "Lambda timeout after 5 minutes"

**Cause:** SageMaker endpoint is slow or agent is stuck

**Solution:**
1. Check SageMaker endpoint latency in CloudWatch
2. Increase Lambda timeout:
   ```typescript
   // In lambda-agent-stack.ts
   timeout: cdk.Duration.minutes(10),
   ```
3. Check agent logs for infinite loops

---

### Issue: "API returns 502 Bad Gateway"

**Cause:** Lambda returned invalid response format

**Solution:** Lambda must return:
```json
{
  "statusCode": 200,
  "headers": {"Content-Type": "application/json"},
  "body": "{\"answer\":\"...\"}"
}
```

Check Lambda logs for errors.

---

## Cleanup (Delete Everything)

**WARNING:** This will delete all resources and stop incurring charges.

```bash
# Delete all stacks (takes ~10 minutes)
cdk destroy --all

# Confirm each deletion when prompted
```

**What gets deleted:**
- ✅ SageMaker endpoint (stops billing)
- ✅ Lambda function
- ✅ API Gateway
- ✅ CloudWatch dashboards + alarms
- ⚠️ Secrets (NOT deleted - manual cleanup required)
- ⚠️ CloudWatch Logs (NOT deleted - manual cleanup required)
- ⚠️ ECR images (NOT deleted - manual cleanup required)

**Manual cleanup:**
```bash
# Delete secrets
aws secretsmanager delete-secret --secret-id langgraph/tavily-api-key --force-delete-without-recovery

# Delete log groups
aws logs delete-log-group --log-group-name /aws/lambda/langgraph-agent
aws logs delete-log-group --log-group-name /aws/apigateway/langgraph-agent

# Delete ECR images
aws ecr batch-delete-image \
  --repository-name cdk-<hash>-container-assets-<account>-us-east-1 \
  --image-ids imageTag=latest
```

---

## Advanced Customization

### Using a Different Model

Edit `bin/app.ts`:

```typescript
const sagemakerStack = new SagemakerEndpointStack(app, `${stackPrefix}-sagemaker`, {
  // Change model ID
  modelId: 'huggingface-llm-llama-2-13b-chat-f',  // Instead of Mistral 7B
  // Use larger instance
  instanceType: 'ml.g5.12xlarge',  // Llama 13B needs more VRAM
  ...
});
```

### Adding More Tools

Edit `agent/tools.py` (after we create it), then redeploy:

```bash
cdk deploy langgraph-dev-lambda
```

### Multi-Region Deployment

```bash
# Deploy to eu-west-1
export CDK_DEFAULT_REGION=eu-west-1
cdk deploy --all
```

### CI/CD Integration

Add to your `.github/workflows/deploy.yml`:

```yaml
- name: Deploy to AWS
  run: |
    npm install
    npm run build
    cdk deploy --all --require-approval never
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## Next Steps

After successful deployment:

1. ✅ Confirm SNS email subscription for alarms
2. ✅ Test API with sample questions
3. ✅ Review CloudWatch dashboard
4. ✅ Set up cost budget alerts
5. ✅ Customize agent code in `agent/` directory
6. ✅ Add more tools or change prompts
7. ✅ Integrate with your application (web app, mobile app, etc.)

---

## Support

For issues with:
- **CDK deployment:** Check CloudFormation console for detailed error messages
- **SageMaker endpoint:** Check SageMaker console → Endpoints → Events
- **Lambda function:** Check CloudWatch Logs
- **Agent code:** See `agent/README.md` for debugging

For questions about AWS CDK: https://docs.aws.amazon.com/cdk/

For questions about this repository: File an issue on GitHub
