import * as cdk from 'aws-cdk-lib';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

/**
 * API Gateway Stack
 *
 * What this does:
 * - Creates a REST API that users can call to interact with the agent
 * - Routes HTTP requests to the Lambda function
 * - Handles authentication, rate limiting, CORS, logging
 *
 * For AI/ML Scientists:
 * - API Gateway is like Flask/FastAPI, but fully managed by AWS
 * - You define routes (POST /ask), it handles scaling/security/monitoring
 * - Users send: curl -X POST https://your-api.amazonaws.com/ask -d '{"question":"..."}'
 * - API Gateway triggers your Lambda, returns the response
 *
 * Why API Gateway instead of Lambda Function URL:
 * - Built-in rate limiting (prevents abuse)
 * - API keys for tracking usage per user
 * - Request/response transformation
 * - CORS support for web apps
 * - Integration with WAF (Web Application Firewall)
 *
 * Cost: $3.50 per million requests (first 1B requests)
 *       Essentially free for small-scale usage
 */

export interface ApiGatewayStackProps extends cdk.StackProps {
  /** The Lambda function to integrate with */
  lambdaFunction: lambda.IFunction;
}

export class ApiGatewayStack extends cdk.Stack {
  /** The API Gateway REST API */
  public readonly api: apigateway.RestApi;

  /** The API endpoint URL */
  public readonly apiUrl: string;

  constructor(scope: Construct, id: string, props: ApiGatewayStackProps) {
    super(scope, id, props);

    /**
     * STEP 1: Create CloudWatch Log Group for API Access Logs
     *
     * Access logs record every request:
     * - Who called the API (IP address)
     * - What endpoint ($context.resourcePath)
     * - When ($context.requestTime)
     * - Response status ($context.status)
     * - Latency ($context.responseLatency)
     *
     * Useful for debugging, auditing, and usage analytics
     */
    const accessLogGroup = new logs.LogGroup(this, 'ApiAccessLogs', {
      logGroupName: '/aws/apigateway/langgraph-agent',
      retention: logs.RetentionDays.ONE_WEEK,  // 7 days for dev, 30+ for prod
      removalPolicy: cdk.RemovalPolicy.DESTROY,  // Delete logs when stack is destroyed
    });

    /**
     * STEP 2: Create REST API
     *
     * REST API vs HTTP API:
     * - REST API: More features (API keys, usage plans, request validation)
     * - HTTP API: Cheaper, simpler (good for internal APIs)
     * We use REST API for the additional features
     */
    this.api = new apigateway.RestApi(this, 'LangGraphApi', {
      restApiName: 'LangGraph Agent API',
      description: 'API for interacting with LangGraph agent on SageMaker',

      /**
       * Deployment options
       *
       * stageName: Environment name (dev, staging, prod)
       * - Creates URL: https://{api-id}.execute-api.{region}.amazonaws.com/{stageName}
       * - Example: https://abc123.execute-api.us-east-1.amazonaws.com/prod
       */
      deployOptions: {
        stageName: 'prod',
        // Enable access logging
        accessLogDestination: new apigateway.LogGroupLogDestination(accessLogGroup),
        accessLogFormat: apigateway.AccessLogFormat.jsonWithStandardFields({
          caller: true,
          httpMethod: true,
          ip: true,
          protocol: true,
          requestTime: true,
          resourcePath: true,
          responseLength: true,
          status: true,
          user: true,
        }),
        // Enable execution logging (logs Lambda integration details)
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true,  // Log full request/response bodies (disable in prod for security)
        // Enable metrics
        metricsEnabled: true,
        // Throttling (rate limiting)
        throttlingRateLimit: 100,  // Requests per second
        throttlingBurstLimit: 200,  // Burst capacity
      },

      /**
       * CORS (Cross-Origin Resource Sharing)
       *
       * If you're calling this API from a web browser, you need CORS
       * This allows your frontend (e.g., React app) to call the API
       *
       * For AI/ML Scientists:
       * - Browsers block requests to different domains for security
       * - CORS tells the browser "this domain is allowed"
       * - If you're only using curl/Python, you don't need this
       */
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,  // For dev; in prod, specify exact domains
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: [
          'Content-Type',
          'X-Amz-Date',
          'Authorization',
          'X-Api-Key',
          'X-Amz-Security-Token',
        ],
        allowCredentials: true,
      },

      /**
       * CloudWatch role
       *
       * API Gateway needs permission to write logs to CloudWatch
       */
      cloudWatchRole: true,
    });

    /**
     * STEP 3: Create Lambda Integration
     *
     * This connects API Gateway → Lambda
     * When a request comes in, API Gateway invokes the Lambda function
     */
    const lambdaIntegration = new apigateway.LambdaIntegration(props.lambdaFunction, {
      /**
       * Proxy integration (recommended)
       *
       * true = API Gateway passes the raw request to Lambda
       * Lambda returns a response with statusCode, headers, body
       *
       * false = You define request/response mappings in API Gateway
       * (more complex, only needed for legacy systems)
       */
      proxy: true,

      /**
       * Request templates (optional)
       *
       * Transform the request before sending to Lambda
       * Example: Extract just the body and pass it to Lambda
       *
       * requestTemplates: {
       *   'application/json': '{"question": $input.json("$.question")}',
       * },
       */

      /**
       * Integration responses (if proxy=false)
       *
       * Map Lambda response to HTTP response
       * With proxy=true, this is automatic
       */
    });

    /**
     * STEP 4: Define API Resources and Methods
     *
     * Resource structure:
     * /
     * ├── /ask (POST)
     * ├── /health (GET)
     * └── /version (GET)
     */

    // POST /ask - Main endpoint for asking questions
    const askResource = this.api.root.addResource('ask');
    askResource.addMethod('POST', lambdaIntegration, {
      /**
       * Request validation
       *
       * Validate request before invoking Lambda (saves Lambda invocations)
       * - Request body must match the schema
       * - Missing required fields → 400 Bad Request (no Lambda invocation)
       */
      requestValidator: new apigateway.RequestValidator(this, 'AskRequestValidator', {
        restApi: this.api,
        validateRequestBody: true,
        validateRequestParameters: false,
      }),
      requestModels: {
        'application/json': new apigateway.Model(this, 'AskRequestModel', {
          restApi: this.api,
          contentType: 'application/json',
          description: 'Request model for /ask endpoint',
          schema: {
            type: apigateway.JsonSchemaType.OBJECT,
            required: ['question'],
            properties: {
              question: {
                type: apigateway.JsonSchemaType.STRING,
                minLength: 1,
                maxLength: 2000,
                description: 'The question to ask the agent',
              },
              chat_history: {
                type: apigateway.JsonSchemaType.ARRAY,
                description: 'Optional conversation history',
                items: {
                  type: apigateway.JsonSchemaType.OBJECT,
                  properties: {
                    role: { type: apigateway.JsonSchemaType.STRING },
                    content: { type: apigateway.JsonSchemaType.STRING },
                  },
                },
              },
            },
          },
        }),
      },
      /**
       * API Key required (optional)
       *
       * Force users to include x-api-key header
       * Useful for tracking usage per user
       */
      // apiKeyRequired: true,

      /**
       * Method responses (documentation only with proxy integration)
       */
      methodResponses: [
        {
          statusCode: '200',
          responseModels: {
            'application/json': apigateway.Model.EMPTY_MODEL,
          },
        },
        {
          statusCode: '400',
          responseModels: {
            'application/json': apigateway.Model.ERROR_MODEL,
          },
        },
        {
          statusCode: '500',
          responseModels: {
            'application/json': apigateway.Model.ERROR_MODEL,
          },
        },
      ],
    });

    // GET /health - Health check endpoint
    const healthResource = this.api.root.addResource('health');
    healthResource.addMethod('GET', new apigateway.MockIntegration({
      // Mock integration = no Lambda, API Gateway responds directly
      integrationResponses: [{
        statusCode: '200',
        responseTemplates: {
          'application/json': '{"status": "healthy", "timestamp": "$context.requestTime"}',
        },
      }],
      passthroughBehavior: apigateway.PassthroughBehavior.NEVER,
      requestTemplates: {
        'application/json': '{"statusCode": 200}',
      },
    }), {
      methodResponses: [{
        statusCode: '200',
      }],
    });

    // GET /version - Version info (useful for deployments)
    const versionResource = this.api.root.addResource('version');
    versionResource.addMethod('GET', new apigateway.MockIntegration({
      integrationResponses: [{
        statusCode: '200',
        responseTemplates: {
          'application/json': `{"version": "1.0.0", "deployed": "${new Date().toISOString()}"}`,
        },
      }],
      passthroughBehavior: apigateway.PassthroughBehavior.NEVER,
      requestTemplates: {
        'application/json': '{"statusCode": 200}',
      },
    }), {
      methodResponses: [{
        statusCode: '200',
      }],
    });

    /**
     * STEP 5: Usage Plan (Optional)
     *
     * Enforce quotas per API key:
     * - 1,000 requests per day
     * - 10 requests per second
     *
     * Uncomment to enable:
     */
    // const usagePlan = this.api.addUsagePlan('UsagePlan', {
    //   name: 'Standard',
    //   throttle: {
    //     rateLimit: 10,  // requests per second
    //     burstLimit: 20,
    //   },
    //   quota: {
    //     limit: 1000,  // requests per day
    //     period: apigateway.Period.DAY,
    //   },
    // });
    // usagePlan.addApiStage({
    //   stage: this.api.deploymentStage,
    // });

    // Create API keys
    // const apiKey = this.api.addApiKey('ApiKey', {
    //   apiKeyName: 'langgraph-api-key',
    // });
    // usagePlan.addApiKey(apiKey);

    /**
     * OUTPUTS
     */
    this.apiUrl = this.api.url;

    new cdk.CfnOutput(this, 'ApiUrl', {
      value: this.apiUrl,
      description: 'API Gateway endpoint URL',
      exportName: `ApiUrl-${this.stackName}`,
    });

    new cdk.CfnOutput(this, 'ApiId', {
      value: this.api.restApiId,
      description: 'API Gateway ID',
    });

    new cdk.CfnOutput(this, 'AskEndpoint', {
      value: `${this.apiUrl}ask`,
      description: 'Full URL for /ask endpoint',
    });

    new cdk.CfnOutput(this, 'CurlExample', {
      value: `curl -X POST ${this.apiUrl}ask -H "Content-Type: application/json" -d '{"question":"What is the capital of France?"}'`,
      description: 'Example curl command to test the API',
    });

    new cdk.CfnOutput(this, 'PythonExample', {
      value: `import requests; response = requests.post("${this.apiUrl}ask", json={"question": "What is 2+2?"}); print(response.json())`,
      description: 'Example Python code to call the API',
    });
  }
}
