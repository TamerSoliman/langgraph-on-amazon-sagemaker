import * as cdk from 'aws-cdk-lib';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

/**
 * SageMaker Endpoint Stack
 *
 * What this does:
 * - Deploys a Large Language Model (LLM) as a SageMaker real-time inference endpoint
 * - Creates: Model → Endpoint Configuration → Endpoint
 * - The endpoint is a managed GPU instance that serves model predictions via API
 *
 * For AI/ML Scientists:
 * - This is similar to model.to('cuda') in PyTorch, but:
 *   * Model is hosted on AWS-managed GPU instances
 *   * Accessible via HTTPS API (no need for VPN to lab server)
 *   * Auto-scaling, health checks, and monitoring included
 *   * Costs ~$723/month for ml.g5.xlarge (vs buying your own GPU)
 *
 * Architecture:
 * ┌─────────────────┐
 * │ SageMaker Model │ ← Points to Docker image + model weights
 * └────────┬────────┘
 *          │
 *          ▼
 * ┌──────────────────────────┐
 * │ Endpoint Configuration   │ ← Specifies instance type, count, etc.
 * └────────┬─────────────────┘
 *          │
 *          ▼
 * ┌──────────────────────────┐
 * │ Endpoint (Running GPU)   │ ← Accepts inference requests
 * └──────────────────────────┘
 */

export interface SagemakerEndpointStackProps extends cdk.StackProps {
  /**
   * Model ID from SageMaker JumpStart
   * Examples:
   * - 'huggingface-llm-mistral-7b-instruct' (default)
   * - 'huggingface-llm-llama-2-13b-chat-f'
   * - 'huggingface-llm-falcon-7b-instruct-bf16'
   */
  modelId: string;

  /**
   * EC2 instance type for hosting the model
   * - ml.g5.xlarge: 1x NVIDIA A10G GPU, 24GB VRAM, $1.006/hour
   * - ml.g5.2xlarge: 1x A10G, more CPU/RAM, $1.515/hour
   * - ml.g5.12xlarge: 4x A10G, 96GB VRAM, $7.09/hour (for larger models)
   *
   * Rule of thumb: Model size × 1.2 ≤ GPU VRAM
   * Mistral 7B in FP16: ~14GB → fits on ml.g5.xlarge (24GB)
   */
  instanceType: string;
}

export class SagemakerEndpointStack extends cdk.Stack {
  /** The name of the deployed endpoint (other stacks will reference this) */
  public readonly endpointName: string;

  /** The endpoint itself (for adding dependencies) */
  public readonly endpoint: sagemaker.CfnEndpoint;

  constructor(scope: Construct, id: string, props: SagemakerEndpointStackProps) {
    super(scope, id, props);

    /**
     * STEP 1: IAM Role for SageMaker
     *
     * Why needed: SageMaker needs permissions to:
     * - Pull Docker images from ECR (Elastic Container Registry)
     * - Download model weights from S3
     * - Write logs to CloudWatch
     * - Access AWS resources on your behalf
     *
     * For AI/ML Scientists:
     * - IAM roles are like Unix permissions, but for AWS services
     * - This role is what SageMaker "assumes" when running your model
     * - Least privilege principle: only grant what's needed
     */
    const sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: 'Execution role for SageMaker endpoint',
      managedPolicies: [
        // Pre-built AWS policy with common SageMaker permissions
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    /**
     * STEP 2: Get Model Artifacts from JumpStart
     *
     * IMPORTANT: This is a simplified example using a placeholder.
     * In reality, you need to:
     *
     * Option A (Recommended): Deploy via JumpStart UI first, then import:
     *   const endpoint = sagemaker.CfnEndpoint.fromEndpointName(
     *     this, 'ImportedEndpoint', 'jumpstart-dft-hf-llm-mistral-7b-instruct'
     *   );
     *
     * Option B: Use SageMaker Python SDK to get model artifact URIs:
     *   from sagemaker.jumpstart.model import JumpStartModel
     *   model = JumpStartModel(model_id='huggingface-llm-mistral-7b-instruct')
     *   image_uri = model.image_uri
     *   model_data = model.model_data
     *
     * Option C: Full CDK deployment (complex - requires knowing exact URIs)
     *
     * For this template, we'll use Option C with hardcoded URIs for Mistral 7B.
     * These URIs are region-specific and version-specific!
     */

    // Model artifact URI (S3 location of model weights)
    // This is a placeholder - replace with actual JumpStart model URI
    const modelDataUrl = `s3://jumpstart-cache-prod-${this.region}/huggingface-infer/prepack/v1.0.0/infer-prepack-huggingface-llm-mistral-7b-instruct.tar.gz`;

    // Container image URI (Docker image with serving code)
    // JumpStart uses HuggingFace TGI (Text Generation Inference) containers
    const imageUri = `763104351884.dkr.ecr.${this.region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04`;

    /**
     * STEP 3: Create SageMaker Model
     *
     * This defines WHAT to deploy (container + model weights)
     * but doesn't actually deploy it yet
     */
    const model = new sagemaker.CfnModel(this, 'MistralModel', {
      modelName: `langgraph-${props.modelId}-${cdk.Names.uniqueId(this).slice(-8)}`,
      executionRoleArn: sagemakerRole.roleArn,
      primaryContainer: {
        image: imageUri,
        modelDataUrl: modelDataUrl,
        environment: {
          // HuggingFace TGI container environment variables
          'SAGEMAKER_PROGRAM': 'inference.py',
          'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
          'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
          'SAGEMAKER_REGION': this.region,
          // Model-specific settings
          'HF_MODEL_ID': '/opt/ml/model',
          'MAX_INPUT_LENGTH': '2048',
          'MAX_TOTAL_TOKENS': '4096',
          'SM_NUM_GPUS': '1',
        },
      },
    });

    /**
     * STEP 4: Create Endpoint Configuration
     *
     * This defines HOW to deploy (instance type, count, etc.)
     *
     * For AI/ML Scientists:
     * - Initial instance count: Number of GPU instances (1 = single endpoint)
     * - Variant: Enables A/B testing (deploy model_v1 and model_v2 simultaneously,
     *   route 90% traffic to v1, 10% to v2, compare metrics)
     * - Initial weight: Traffic distribution (100 = all traffic to this variant)
     */
    const endpointConfig = new sagemaker.CfnEndpointConfig(this, 'EndpointConfig', {
      endpointConfigName: `langgraph-${props.modelId}-config-${cdk.Names.uniqueId(this).slice(-8)}`,
      productionVariants: [
        {
          modelName: model.attrModelName,
          variantName: 'AllTraffic',
          initialInstanceCount: 1,
          instanceType: props.instanceType,
          initialVariantWeight: 100,
          // Optional: Configure auto-scaling
          // containerStartupHealthCheckTimeoutInSeconds: 600,
        },
      ],
      // Optional: Data capture for monitoring model drift
      // dataCaptureConfig: {
      //   enableCapture: true,
      //   initialSamplingPercentage: 100,
      //   destinationS3Uri: `s3://my-bucket/sagemaker-data-capture`,
      //   captureOptions: [
      //     { captureMode: 'Input' },
      //     { captureMode: 'Output' },
      //   ],
      // },
    });

    endpointConfig.addDependency(model);

    /**
     * STEP 5: Create Endpoint (Actual Deployment)
     *
     * This launches the GPU instance and starts serving the model
     * Takes 5-10 minutes to complete!
     *
     * Status progression:
     * Creating → InService (ready) or Failed (check CloudWatch logs)
     */
    this.endpoint = new sagemaker.CfnEndpoint(this, 'Endpoint', {
      endpointName: `langgraph-${props.modelId}-endpoint`,
      endpointConfigName: endpointConfig.attrEndpointConfigName,
      tags: [
        { key: 'Model', value: props.modelId },
        { key: 'InstanceType', value: props.instanceType },
        { key: 'Framework', value: 'HuggingFace-TGI' },
      ],
    });

    this.endpoint.addDependency(endpointConfig);

    // Store endpoint name for other stacks
    this.endpointName = this.endpoint.attrEndpointName;

    /**
     * OUTPUTS
     */
    new cdk.CfnOutput(this, 'EndpointName', {
      value: this.endpointName,
      description: 'SageMaker endpoint name',
      exportName: `SageMakerEndpointName-${this.stackName}`,
    });

    new cdk.CfnOutput(this, 'EndpointArn', {
      value: this.endpoint.ref,
      description: 'SageMaker endpoint ARN',
    });

    new cdk.CfnOutput(this, 'ModelName', {
      value: model.attrModelName,
      description: 'SageMaker model name',
    });

    new cdk.CfnOutput(this, 'EstimatedMonthlyCost', {
      value: this.calculateMonthlyCost(props.instanceType),
      description: 'Estimated monthly cost (USD) for 24/7 operation',
    });

    /**
     * COST MONITORING
     *
     * Add a CloudWatch alarm if monthly cost exceeds threshold
     */
    const monthlyCost = parseFloat(this.calculateMonthlyCost(props.instanceType).replace('$', ''));
    if (monthlyCost > 500) {
      new cdk.CfnOutput(this, 'CostWarning', {
        value: `⚠️  WARNING: This endpoint costs ${this.calculateMonthlyCost(props.instanceType)}/month. Consider ml.g5.xlarge for dev/testing.`,
        description: 'Cost warning',
      });
    }
  }

  /**
   * Calculate estimated monthly cost based on instance type
   *
   * Prices as of 2024 (us-east-1):
   * - ml.g5.xlarge: $1.006/hour × 730 hours = $734/month
   * - ml.g5.2xlarge: $1.515/hour × 730 hours = $1,106/month
   * - ml.g5.4xlarge: $2.03/hour × 730 hours = $1,482/month
   */
  private calculateMonthlyCost(instanceType: string): string {
    const hourlyRates: { [key: string]: number } = {
      'ml.g5.xlarge': 1.006,
      'ml.g5.2xlarge': 1.515,
      'ml.g5.4xlarge': 2.03,
      'ml.g5.12xlarge': 7.09,
      'ml.g5.24xlarge': 10.18,
      'ml.g5.48xlarge': 20.36,
    };

    const hourlyRate = hourlyRates[instanceType] || 1.0; // Default fallback
    const monthlyCost = hourlyRate * 730; // 730 = average hours per month

    return `$${monthlyCost.toFixed(2)}`;
  }
}
