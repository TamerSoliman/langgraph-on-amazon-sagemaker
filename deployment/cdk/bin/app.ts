#!/usr/bin/env node

/**
 * AWS CDK Application Entry Point
 *
 * This file orchestrates the deployment of the complete LangGraph on SageMaker stack.
 *
 * For AI/ML Scientists:
 * - CDK (Cloud Development Kit) is "infrastructure as code" - you write TypeScript/Python
 *   to define AWS resources instead of clicking through the AWS console
 * - This is like a "recipe" that AWS follows to set up all the cloud resources
 * - Think of it as a Dockerfile for cloud infrastructure
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { SecretsStack } from '../lib/secrets-stack';
import { SagemakerEndpointStack } from '../lib/sagemaker-endpoint-stack';
import { LambdaAgentStack } from '../lib/lambda-agent-stack';
import { ApiGatewayStack } from '../lib/api-gateway-stack';
import { MonitoringStack } from '../lib/monitoring-stack';

const app = new cdk.App();

// Get environment from context (dev, staging, prod)
// Usage: cdk deploy --context environment=prod
const environment = app.node.tryGetContext('environment') || 'dev';

// AWS account and region from environment or defaults
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
};

console.log(`Deploying to environment: ${environment}`);
console.log(`Account: ${env.account}, Region: ${env.region}`);

// Stack naming convention: langgraph-{environment}-{component}
const stackPrefix = `langgraph-${environment}`;

/**
 * STACK 1: Secrets Management
 *
 * What it does: Stores sensitive data (API keys) securely in AWS Secrets Manager
 * Why first: Other stacks need to reference these secrets
 *
 * For AI/ML Scientists:
 * - Never hardcode API keys in code! Use AWS Secrets Manager
 * - This is like environment variables, but encrypted and access-controlled
 * - Cost: ~$0.40/month per secret
 */
const secretsStack = new SecretsStack(app, `${stackPrefix}-secrets`, {
  env,
  stackName: `${stackPrefix}-secrets`,
  description: 'Secrets management for LangGraph agent (Tavily API key, etc.)',
  tags: {
    Environment: environment,
    Project: 'langgraph-sagemaker',
    Component: 'secrets',
  },
});

/**
 * STACK 2: SageMaker Endpoint
 *
 * What it does: Deploys the Mistral 7B LLM model as a SageMaker endpoint
 * Why: This is the GPU-powered "brain" that generates text
 *
 * For AI/ML Scientists:
 * - This is similar to loading a model in PyTorch, but on managed infrastructure
 * - SageMaker handles: model serving, auto-scaling, health checks, A/B testing
 * - You just call an API, no need to manage GPU instances directly
 * - Cost: ~$723/month for ml.g5.xlarge (24/7 operation)
 *
 * Alternative: Use SageMaker JumpStart UI (easier but not reproducible)
 */
const sagemakerStack = new SagemakerEndpointStack(app, `${stackPrefix}-sagemaker`, {
  env,
  stackName: `${stackPrefix}-sagemaker`,
  description: 'SageMaker endpoint for Mistral 7B Instruct model',
  modelId: 'huggingface-llm-mistral-7b-instruct',  // JumpStart model ID
  instanceType: environment === 'prod' ? 'ml.g5.2xlarge' : 'ml.g5.xlarge',
  tags: {
    Environment: environment,
    Project: 'langgraph-sagemaker',
    Component: 'llm-endpoint',
  },
});

/**
 * STACK 3: Lambda Agent
 *
 * What it does: Deploys the LangGraph agent code as a Lambda function
 * Why: Serverless = no servers to manage, auto-scaling, pay-per-use
 *
 * For AI/ML Scientists:
 * - Lambda is like running a Python script in the cloud, triggered by events
 * - Your LangGraph code runs here (the orchestration logic)
 * - It calls the SageMaker endpoint when it needs LLM inference
 * - Cost: ~$0.35/month for 1,000 questions/day (super cheap!)
 *
 * Alternative: ECS/Fargate for always-on container (~$15/month)
 */
const lambdaStack = new LambdaAgentStack(app, `${stackPrefix}-lambda`, {
  env,
  stackName: `${stackPrefix}-lambda`,
  description: 'Lambda function running LangGraph agent',
  sagemakerEndpointName: sagemakerStack.endpointName,
  tavilySecretArn: secretsStack.tavilySecretArn,
  tags: {
    Environment: environment,
    Project: 'langgraph-sagemaker',
    Component: 'agent',
  },
});

// Lambda needs to wait for SageMaker endpoint to be ready
lambdaStack.addDependency(sagemakerStack);
lambdaStack.addDependency(secretsStack);

/**
 * STACK 4: API Gateway
 *
 * What it does: Creates a REST API that users can call to ask questions
 * Why: Provides a public HTTPS endpoint for your agent
 *
 * For AI/ML Scientists:
 * - API Gateway is like Flask/FastAPI, but fully managed by AWS
 * - Users POST questions to https://your-api.amazonaws.com/ask
 * - API Gateway triggers your Lambda function
 * - Includes: rate limiting, API keys, CORS, logging
 * - Cost: $3.50 per million requests (essentially free for small usage)
 */
const apiStack = new ApiGatewayStack(app, `${stackPrefix}-api`, {
  env,
  stackName: `${stackPrefix}-api`,
  description: 'API Gateway for LangGraph agent',
  lambdaFunction: lambdaStack.agentFunction,
  tags: {
    Environment: environment,
    Project: 'langgraph-sagemaker',
    Component: 'api',
  },
});

apiStack.addDependency(lambdaStack);

/**
 * STACK 5: Monitoring & Alarms
 *
 * What it does: Sets up CloudWatch dashboards and alarms
 * Why: Know when things break before users complain!
 *
 * For AI/ML Scientists:
 * - CloudWatch is like TensorBoard for infrastructure (logs, metrics, dashboards)
 * - Monitors: error rates, latency, cost, LLM endpoint health
 * - Alarms: send email/Slack when errors spike or budget exceeded
 * - Cost: ~$3/month for dashboards + $0.10/alarm
 */
const monitoringStack = new MonitoringStack(app, `${stackPrefix}-monitoring`, {
  env,
  stackName: `${stackPrefix}-monitoring`,
  description: 'Monitoring and alarms for LangGraph agent',
  lambdaFunction: lambdaStack.agentFunction,
  sagemakerEndpoint: sagemakerStack.endpointName,
  apiGateway: apiStack.api,
  alarmEmail: process.env.ALARM_EMAIL || 'your-email@example.com',
  tags: {
    Environment: environment,
    Project: 'langgraph-sagemaker',
    Component: 'monitoring',
  },
});

monitoringStack.addDependency(apiStack);
monitoringStack.addDependency(sagemakerStack);

/**
 * Output Summary
 *
 * After deployment, CDK will print these values to the console.
 * You'll need them to use your agent!
 */
new cdk.CfnOutput(app, 'ApiEndpoint', {
  value: apiStack.apiUrl,
  description: 'API Gateway endpoint URL',
  exportName: `${stackPrefix}-api-url`,
});

new cdk.CfnOutput(app, 'SageMakerEndpoint', {
  value: sagemakerStack.endpointName,
  description: 'SageMaker endpoint name',
  exportName: `${stackPrefix}-sagemaker-endpoint`,
});

new cdk.CfnOutput(app, 'DashboardUrl', {
  value: `https://console.aws.amazon.com/cloudwatch/home?region=${env.region}#dashboards:name=${stackPrefix}-dashboard`,
  description: 'CloudWatch Dashboard URL',
});

// Tags applied to all resources
cdk.Tags.of(app).add('ManagedBy', 'CDK');
cdk.Tags.of(app).add('Project', 'langgraph-sagemaker');
cdk.Tags.of(app).add('CostCenter', 'ML-Engineering');
