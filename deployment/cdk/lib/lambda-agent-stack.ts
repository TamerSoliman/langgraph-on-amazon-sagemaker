import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';
import * as path from 'path';

/**
 * Lambda Agent Stack
 *
 * What this does:
 * - Deploys the LangGraph agent code as an AWS Lambda function
 * - Packages Python code + dependencies into a Docker container
 * - Grants permissions to call SageMaker endpoint and read secrets
 *
 * For AI/ML Scientists:
 * - Lambda = run code without managing servers
 * - Your LangGraph code runs when triggered (API call, schedule, event)
 * - Auto-scales: 1 request → 1 Lambda, 1000 requests → 1000 Lambdas
 * - Cost: Pay only for compute time (billed in 1ms increments)
 * - Timeout: Max 15 minutes per invocation
 *
 * Why Lambda for agents:
 * - Agent logic is lightweight (CPU-only: parsing, API calls, graph traversal)
 * - LLM inference happens on SageMaker (GPU)
 * - Lambda → cheap ($0.35/month for 1K questions), SageMaker → expensive ($723/month)
 *
 * Alternative: ECS Fargate if you need >15 min execution or always-on container
 */

export interface LambdaAgentStackProps extends cdk.StackProps {
  /** Name of the SageMaker endpoint to call for LLM inference */
  sagemakerEndpointName: string;

  /** ARN of the Tavily API key secret in Secrets Manager */
  tavilySecretArn: string;
}

export class LambdaAgentStack extends cdk.Stack {
  /** The Lambda function running the LangGraph agent */
  public readonly agentFunction: lambda.DockerImageFunction;

  constructor(scope: Construct, id: string, props: LambdaAgentStackProps) {
    super(scope, id, props);

    /**
     * STEP 1: Create Lambda Execution Role
     *
     * This role defines what the Lambda function can do:
     * - Call SageMaker endpoint (invoke_endpoint)
     * - Read secrets from Secrets Manager (GetSecretValue)
     * - Write logs to CloudWatch (CreateLogGroup, PutLogEvents)
     *
     * For AI/ML Scientists:
     * - Think of this as sudo permissions for your code
     * - Principle of least privilege: only grant what's needed
     * - If Lambda tries something not in this role, it gets "Access Denied"
     */
    const executionRole = new iam.Role(this, 'LambdaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Execution role for LangGraph agent Lambda',
      managedPolicies: [
        // Basic Lambda execution (CloudWatch Logs)
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    /**
     * Grant permission to invoke SageMaker endpoint
     *
     * Action: sagemaker:InvokeEndpoint
     * Resource: The specific endpoint (not all endpoints)
     */
    executionRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['sagemaker:InvokeEndpoint'],
      resources: [
        `arn:aws:sagemaker:${this.region}:${this.account}:endpoint/${props.sagemakerEndpointName}`,
      ],
    }));

    /**
     * Grant permission to read Tavily API key from Secrets Manager
     *
     * Action: secretsmanager:GetSecretValue
     * Resource: The specific secret (not all secrets)
     */
    executionRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [props.tavilySecretArn],
    }));

    /**
     * STEP 2: Define Lambda Function (Docker Container)
     *
     * Why Docker instead of ZIP:
     * - langchain + dependencies > 50MB (ZIP limit is 50MB uncompressed)
     * - Docker allows up to 10GB
     * - Easier to match local dev environment exactly
     *
     * The Dockerfile must exist at: ../../agent/Dockerfile
     * (relative to this CDK project root)
     */
    this.agentFunction = new lambda.DockerImageFunction(this, 'AgentFunction', {
      functionName: 'langgraph-agent',
      description: 'LangGraph agent with SageMaker LLM backend',

      /**
       * Docker image location
       *
       * This points to the agent code directory with Dockerfile
       * CDK will:
       * 1. Build the Docker image
       * 2. Push to Amazon ECR (Elastic Container Registry)
       * 3. Create Lambda from that image
       */
      code: lambda.DockerImageCode.fromImageAsset(
        path.join(__dirname, '../../../agent'),  // Path to agent/ directory
        {
          file: 'Dockerfile',  // Dockerfile name
          // Optional: pass build args
          // buildArgs: {
          //   PYTHON_VERSION: '3.11',
          // },
        }
      ),

      /**
       * Resource allocation
       *
       * Memory: 1024 MB (1GB)
       * - LangGraph + langchain ~200MB
       * - Tavily SDK ~50MB
       * - Runtime overhead ~100MB
       * - Working memory ~600MB (for large contexts)
       *
       * Timeout: 5 minutes (300 seconds)
       * - Typical agent execution: 5-15 seconds
       * - Includes: LLM calls (2-5s each), tool execution (0.5-2s), overhead (1-2s)
       * - Buffer for slow LLM responses or complex multi-step agents
       *
       * Rule of thumb: (# of LLM calls × 5s) + (# of tool calls × 2s) + 10s buffer
       */
      memorySize: 1024,  // MB
      timeout: cdk.Duration.minutes(5),

      /**
       * Environment variables
       *
       * These are accessible via os.environ['VAR_NAME'] in Python
       */
      environment: {
        'SAGEMAKER_ENDPOINT_NAME': props.sagemakerEndpointName,
        'TAVILY_SECRET_ARN': props.tavilySecretArn,
        'AWS_DEFAULT_REGION': this.region,
        'LOG_LEVEL': 'INFO',
        // Python will fetch Tavily key from Secrets Manager using this ARN
      },

      /**
       * Execution role (created above)
       */
      role: executionRole,

      /**
       * Reserved concurrent executions (optional)
       *
       * Limits how many copies of this Lambda can run simultaneously
       * - undefined = no limit (auto-scales to account limit, usually 1000)
       * - 10 = max 10 concurrent executions (protects against cost spikes)
       *
       * For production:
       * - Set this to expected peak concurrency + 20% buffer
       * - Prevents runaway costs from accidental loops or DDoS
       */
      // reservedConcurrentExecutions: 100,

      /**
       * Log retention
       *
       * CloudWatch Logs retention period
       * - 1 day = cheap, good for dev ($0.50/GB stored)
       * - 30 days = production recommended
       * - Never expire = compliance use case (can get expensive)
       */
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    /**
     * STEP 3: Add X-Ray Tracing (Optional)
     *
     * AWS X-Ray traces requests through distributed systems
     * Helpful for debugging: which step is slow? where did it fail?
     *
     * Uncomment to enable:
     */
    // this.agentFunction.addToRolePolicy(new iam.PolicyStatement({
    //   actions: ['xray:PutTraceSegments', 'xray:PutTelemetryRecords'],
    //   resources: ['*'],
    // }));

    /**
     * STEP 4: Configure Dead Letter Queue (Optional, Recommended)
     *
     * If Lambda fails 3 times, send event to SQS queue for manual review
     * Prevents losing user requests due to transient errors
     *
     * Uncomment to enable:
     */
    // const dlq = new sqs.Queue(this, 'AgentDLQ', {
    //   queueName: 'langgraph-agent-dlq',
    //   retentionPeriod: cdk.Duration.days(14),
    // });
    // this.agentFunction.addEventSourceMapping('DLQMapping', {
    //   eventSourceArn: dlq.queueArn,
    //   onFailure: new lambdaEventSources.SqsEventSource(dlq),
    // });

    /**
     * OUTPUTS
     */
    new cdk.CfnOutput(this, 'FunctionName', {
      value: this.agentFunction.functionName,
      description: 'Lambda function name',
      exportName: `LambdaFunctionName-${this.stackName}`,
    });

    new cdk.CfnOutput(this, 'FunctionArn', {
      value: this.agentFunction.functionArn,
      description: 'Lambda function ARN',
    });

    new cdk.CfnOutput(this, 'TestCommand', {
      value: `aws lambda invoke --function-name ${this.agentFunction.functionName} --payload '{"question":"What is 2+2?","chat_history":[]}' response.json && cat response.json`,
      description: 'Command to test the Lambda function',
    });

    new cdk.CfnOutput(this, 'LogsCommand', {
      value: `aws logs tail /aws/lambda/${this.agentFunction.functionName} --follow`,
      description: 'Command to view Lambda logs in real-time',
    });

    /**
     * COST ESTIMATION
     *
     * Lambda pricing (us-east-1):
     * - Requests: $0.20 per 1M requests
     * - Compute: $0.0000166667 per GB-second
     *
     * Example: 1,000 questions/day, 10 seconds each, 1GB memory
     * - Requests: 30,000/month × $0.20/1M = $0.006
     * - Compute: 30,000 × 10s × 1GB × $0.0000166667 = $5.00
     * - Total: ~$5.01/month
     *
     * Compare to: SageMaker endpoint = $723/month
     * Lambda is negligible!
     */
    const requestsPerMonth = 30000;  // Assume 1K/day
    const avgDurationSeconds = 10;
    const memoryGB = this.agentFunction.memorySize / 1024;

    const requestCost = (requestsPerMonth * 0.20) / 1000000;
    const computeCost = requestsPerMonth * avgDurationSeconds * memoryGB * 0.0000166667;
    const totalCost = requestCost + computeCost;

    new cdk.CfnOutput(this, 'EstimatedMonthlyCost', {
      value: `$${totalCost.toFixed(2)} (assuming ${requestsPerMonth.toLocaleString()} requests/month)`,
      description: 'Estimated Lambda cost per month',
    });
  }
}
