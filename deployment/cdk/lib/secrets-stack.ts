import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

/**
 * Secrets Management Stack
 *
 * What this does:
 * - Creates AWS Secrets Manager secrets for sensitive data (API keys)
 * - Secrets are encrypted at rest and access-controlled via IAM
 *
 * For AI/ML Scientists:
 * - Think of this as a secure key-value store for passwords/API keys
 * - Instead of: api_key = "sk-abc123" (BAD - visible in code/logs)
 * - You do: api_key = secrets_manager.get_secret("tavily-key") (GOOD - encrypted)
 * - Only authorized services (your Lambda) can read these secrets
 *
 * Cost: $0.40/month per secret + $0.05 per 10,000 API calls
 */

export interface SecretsStackProps extends cdk.StackProps {
  // No additional props needed for now
}

export class SecretsStack extends cdk.Stack {
  /**
   * The ARN (Amazon Resource Name) of the Tavily API key secret
   * Other stacks will reference this to grant read permissions
   */
  public readonly tavilySecretArn: string;

  constructor(scope: Construct, id: string, props?: SecretsStackProps) {
    super(scope, id, props);

    /**
     * TAVILY API KEY SECRET
     *
     * How to use after deployment:
     * 1. Deploy this stack: cdk deploy langgraph-dev-secrets
     * 2. Manually set the secret value in AWS console:
     *    - Go to Secrets Manager → tavily-api-key → Store a new secret value
     *    - Enter: {"api_key": "tvly-YOUR_ACTUAL_KEY"}
     * 3. Or use AWS CLI:
     *    aws secretsmanager put-secret-value \
     *      --secret-id tavily-api-key \
     *      --secret-string '{"api_key":"tvly-YOUR_KEY"}'
     *
     * Why JSON format: Allows storing multiple related values in one secret
     * Example: {"api_key": "tvly-123", "endpoint": "https://api.tavily.com"}
     */
    const tavilySecret = new secretsmanager.Secret(this, 'TavilyApiKey', {
      secretName: 'langgraph/tavily-api-key',
      description: 'Tavily API key for web search tool',
      // Secret value is NOT set here (security best practice)
      // You must set it manually after deployment
      // This prevents API keys from being stored in source code
    });

    // Export the ARN so other stacks can reference it
    this.tavilySecretArn = tavilySecret.secretArn;

    /**
     * OUTPUTS
     *
     * These values will be printed after deployment
     * You'll need the secret ARN to manually set the value
     */
    new cdk.CfnOutput(this, 'TavilySecretArn', {
      value: tavilySecret.secretArn,
      description: 'ARN of the Tavily API key secret',
      exportName: 'TavilySecretArn',
    });

    new cdk.CfnOutput(this, 'TavilySecretName', {
      value: tavilySecret.secretName,
      description: 'Name of the Tavily API key secret',
    });

    new cdk.CfnOutput(this, 'SetSecretCommand', {
      value: `aws secretsmanager put-secret-value --secret-id ${tavilySecret.secretName} --secret-string '{"api_key":"YOUR_TAVILY_KEY_HERE"}'`,
      description: 'Command to set the Tavily API key',
    });

    /**
     * OPTIONAL: Additional secrets for other tools
     *
     * Uncomment if you add more external services:
     *
     * // OpenAI API key (if using GPT-4 as alternative LLM)
     * const openaiSecret = new secretsmanager.Secret(this, 'OpenAIApiKey', {
     *   secretName: 'langgraph/openai-api-key',
     *   description: 'OpenAI API key for GPT-4',
     * });
     *
     * // Database credentials (if using custom DB tool)
     * const dbSecret = new secretsmanager.Secret(this, 'DatabaseCredentials', {
     *   secretName: 'langgraph/database-credentials',
     *   description: 'RDS database credentials',
     *   generateSecretString: {
     *     secretStringTemplate: JSON.stringify({ username: 'admin' }),
     *     generateStringKey: 'password',
     *     excludePunctuation: true,
     *     includeSpace: false,
     *   },
     * });
     */

    /**
     * AUTOMATIC ROTATION (Optional, Advanced)
     *
     * For production, consider rotating secrets automatically every 30/90 days
     *
     * tavilySecret.addRotationSchedule('RotationSchedule', {
     *   rotationLambda: myRotationFunction,  // You'd need to create this Lambda
     *   automaticallyAfter: cdk.Duration.days(30),
     * });
     *
     * This requires writing a Lambda function that:
     * 1. Generates a new Tavily API key (via Tavily API)
     * 2. Updates the secret in Secrets Manager
     * 3. Validates the new key works
     */
  }
}
