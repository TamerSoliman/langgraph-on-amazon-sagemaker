import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as cloudwatch_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import { Construct } from 'constructs';

/**
 * Monitoring Stack
 *
 * What this does:
 * - Creates CloudWatch dashboards to visualize agent performance
 * - Sets up alarms that trigger when things go wrong
 * - Sends email/Slack notifications via SNS when alarms fire
 *
 * For AI/ML Scientists:
 * - CloudWatch = TensorBoard for infrastructure (metrics, logs, dashboards)
 * - Monitors: error rates, latency, cost, LLM endpoint health
 * - Alarms = automated alerts ("email me if error rate > 5%")
 * - Think of it as unit tests, but for production (continuous validation)
 *
 * Why monitoring matters:
 * - LLMs can fail silently (return gibberish, not errors)
 * - SageMaker endpoints can throttle/crash
 * - User traffic can spike unexpectedly
 * - You want to know BEFORE users complain
 *
 * Cost: ~$3/month for dashboards, $0.10/alarm/month
 */

export interface MonitoringStackProps extends cdk.StackProps {
  /** Lambda function to monitor */
  lambdaFunction: lambda.IFunction;

  /** SageMaker endpoint name */
  sagemakerEndpoint: string;

  /** API Gateway to monitor */
  apiGateway: apigateway.RestApi;

  /** Email address for alarm notifications */
  alarmEmail: string;
}

export class MonitoringStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: MonitoringStackProps) {
    super(scope, id, props);

    /**
     * STEP 1: Create SNS Topic for Alarm Notifications
     *
     * SNS (Simple Notification Service) = publish-subscribe messaging
     * Alarms publish to this topic → SNS sends emails/SMS/Slack messages
     *
     * For AI/ML Scientists:
     * - Think of SNS as a mailing list
     * - When an alarm "fires", it publishes a message to SNS
     * - SNS forwards the message to all subscribers (email, Slack, PagerDuty)
     */
    const alarmTopic = new sns.Topic(this, 'AlarmTopic', {
      displayName: 'LangGraph Agent Alarms',
      topicName: 'langgraph-agent-alarms',
    });

    // Subscribe email address to receive alarm notifications
    alarmTopic.addSubscription(
      new subscriptions.EmailSubscription(props.alarmEmail)
    );

    // Optional: Add Slack webhook
    // alarmTopic.addSubscription(
    //   new subscriptions.UrlSubscription('https://hooks.slack.com/services/YOUR/WEBHOOK/URL')
    // );

    /**
     * STEP 2: Define CloudWatch Metrics
     *
     * Metrics = time-series data points (e.g., error count every minute)
     * AWS services auto-publish metrics; we just reference them
     */

    // Lambda metrics
    const lambdaErrors = props.lambdaFunction.metricErrors({
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const lambdaDuration = props.lambdaFunction.metricDuration({
      statistic: cloudwatch.Stats.AVERAGE,
      period: cdk.Duration.minutes(5),
    });

    const lambdaInvocations = props.lambdaFunction.metricInvocations({
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const lambdaThrottles = props.lambdaFunction.metricThrottles({
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    // API Gateway metrics
    const api4xxErrors = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: '4XXError',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const api5xxErrors = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: '5XXError',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const apiLatency = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: 'Latency',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
      },
      statistic: cloudwatch.Stats.AVERAGE,
      period: cdk.Duration.minutes(5),
    });

    const apiCount = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: 'Count',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    // SageMaker endpoint metrics
    const sagemakerInvocations = new cloudwatch.Metric({
      namespace: 'AWS/SageMaker',
      metricName: 'ModelInvocations',
      dimensionsMap: {
        EndpointName: props.sagemakerEndpoint,
        VariantName: 'AllTraffic',
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const sagemakerLatency = new cloudwatch.Metric({
      namespace: 'AWS/SageMaker',
      metricName: 'ModelLatency',
      dimensionsMap: {
        EndpointName: props.sagemakerEndpoint,
        VariantName: 'AllTraffic',
      },
      statistic: cloudwatch.Stats.AVERAGE,
      period: cdk.Duration.minutes(5),
    });

    const sagemaker4xxErrors = new cloudwatch.Metric({
      namespace: 'AWS/SageMaker',
      metricName: 'Invocation4XXErrors',
      dimensionsMap: {
        EndpointName: props.sagemakerEndpoint,
        VariantName: 'AllTraffic',
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    const sagemaker5xxErrors = new cloudwatch.Metric({
      namespace: 'AWS/SageMaker',
      metricName: 'Invocation5XXErrors',
      dimensionsMap: {
        EndpointName: props.sagemakerEndpoint,
        VariantName: 'AllTraffic',
      },
      statistic: cloudwatch.Stats.SUM,
      period: cdk.Duration.minutes(5),
    });

    /**
     * STEP 3: Create CloudWatch Dashboard
     *
     * Dashboard = visual representation of metrics (like Grafana)
     * Widgets = charts showing specific metrics
     */
    const dashboard = new cloudwatch.Dashboard(this, 'AgentDashboard', {
      dashboardName: 'langgraph-agent-dashboard',
    });

    // Row 1: Overview metrics
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'API Gateway Requests',
        left: [apiCount],
        width: 12,
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Invocations',
        left: [lambdaInvocations],
        right: [lambdaErrors],
        width: 12,
      })
    );

    // Row 2: Error rates
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'API Gateway Errors',
        left: [api4xxErrors, api5xxErrors],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Errors & Throttles',
        left: [lambdaErrors, lambdaThrottles],
        width: 8,
      }),
      new cloudwatch.GraphWidget({
        title: 'SageMaker Errors',
        left: [sagemaker4xxErrors, sagemaker5xxErrors],
        width: 8,
      })
    );

    // Row 3: Latency metrics
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'API Gateway Latency',
        left: [apiLatency],
        width: 12,
      }),
      new cloudwatch.GraphWidget({
        title: 'SageMaker Model Latency',
        left: [sagemakerLatency],
        width: 12,
      })
    );

    // Row 4: Lambda detailed metrics
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Duration',
        left: [lambdaDuration],
        width: 12,
      }),
      new cloudwatch.SingleValueWidget({
        title: 'Current Lambda Concurrency',
        metrics: [
          props.lambdaFunction.metricConcurrentExecutions({
            statistic: cloudwatch.Stats.MAXIMUM,
          }),
        ],
        width: 6,
      }),
      new cloudwatch.SingleValueWidget({
        title: 'Lambda Error Rate',
        metrics: [
          new cloudwatch.MathExpression({
            expression: '(errors / invocations) * 100',
            usingMetrics: {
              errors: lambdaErrors,
              invocations: lambdaInvocations,
            },
          }),
        ],
        width: 6,
      })
    );

    /**
     * STEP 4: Create Alarms
     *
     * Alarm = threshold-based alert
     * If metric crosses threshold → alarm fires → SNS notification sent
     *
     * Alarm states:
     * - OK: Metric within threshold
     * - ALARM: Metric breached threshold
     * - INSUFFICIENT_DATA: Not enough data to evaluate
     */

    // ALARM 1: High Lambda error rate
    const lambdaErrorAlarm = new cloudwatch.Alarm(this, 'LambdaErrorAlarm', {
      alarmName: 'langgraph-lambda-high-errors',
      alarmDescription: 'Lambda function has high error rate (>5% for 10 minutes)',
      metric: new cloudwatch.MathExpression({
        expression: '(errors / invocations) * 100',
        usingMetrics: {
          errors: lambdaErrors,
          invocations: lambdaInvocations,
        },
      }),
      threshold: 5,  // 5% error rate
      evaluationPeriods: 2,  // 2 consecutive periods (10 minutes total)
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    lambdaErrorAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // ALARM 2: SageMaker endpoint errors
    const sagemakerErrorAlarm = new cloudwatch.Alarm(this, 'SageMakerErrorAlarm', {
      alarmName: 'langgraph-sagemaker-errors',
      alarmDescription: 'SageMaker endpoint has errors',
      metric: sagemaker5xxErrors,
      threshold: 5,  // More than 5 errors in 5 minutes
      evaluationPeriods: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    sagemakerErrorAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // ALARM 3: High API latency
    const apiLatencyAlarm = new cloudwatch.Alarm(this, 'ApiLatencyAlarm', {
      alarmName: 'langgraph-api-high-latency',
      alarmDescription: 'API Gateway latency is high (>30 seconds for 10 minutes)',
      metric: apiLatency,
      threshold: 30000,  // 30 seconds in milliseconds
      evaluationPeriods: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    apiLatencyAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // ALARM 4: Lambda throttling (capacity issue)
    const lambdaThrottleAlarm = new cloudwatch.Alarm(this, 'LambdaThrottleAlarm', {
      alarmName: 'langgraph-lambda-throttled',
      alarmDescription: 'Lambda function is being throttled (hitting concurrency limit)',
      metric: lambdaThrottles,
      threshold: 10,  // More than 10 throttles in 5 minutes
      evaluationPeriods: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    lambdaThrottleAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    // ALARM 5: Anomaly detection (advanced - learns normal patterns)
    // const lambdaDurationAnomalyAlarm = new cloudwatch.Alarm(this, 'LambdaDurationAnomaly', {
    //   alarmName: 'langgraph-lambda-duration-anomaly',
    //   alarmDescription: 'Lambda duration is outside normal bounds',
    //   metric: lambdaDuration,
    //   threshold: 2,  // 2 standard deviations
    //   evaluationPeriods: 2,
    //   comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    //   treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    // });

    /**
     * STEP 5: Cost Monitoring (Optional but Recommended)
     *
     * Monitor estimated daily cost to prevent budget overruns
     *
     * Requires enabling Cost Explorer and creating a budget
     */
    // const costAlarm = new cloudwatch.Alarm(this, 'DailyCostAlarm', {
    //   alarmName: 'langgraph-daily-cost-exceeded',
    //   alarmDescription: 'Daily cost exceeds $50',
    //   metric: new cloudwatch.Metric({
    //     namespace: 'AWS/Billing',
    //     metricName: 'EstimatedCharges',
    //     dimensionsMap: {
    //       Currency: 'USD',
    //     },
    //     statistic: cloudwatch.Stats.MAXIMUM,
    //     period: cdk.Duration.hours(6),
    //   }),
    //   threshold: 50,
    //   evaluationPeriods: 1,
    //   comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    // });
    // costAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(alarmTopic));

    /**
     * OUTPUTS
     */
    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://console.aws.amazon.com/cloudwatch/home?region=${this.region}#dashboards:name=${dashboard.dashboardName}`,
      description: 'CloudWatch Dashboard URL',
    });

    new cdk.CfnOutput(this, 'AlarmTopicArn', {
      value: alarmTopic.topicArn,
      description: 'SNS topic ARN for alarms',
    });

    new cdk.CfnOutput(this, 'SubscriptionNote', {
      value: `Check your email (${props.alarmEmail}) and confirm the SNS subscription to receive alarm notifications`,
      description: 'Important: Confirm SNS subscription',
    });

    /**
     * CUSTOM METRICS (Optional)
     *
     * You can publish custom metrics from your Lambda code:
     *
     * Python example:
     * import boto3
     * cloudwatch = boto3.client('cloudwatch')
     * cloudwatch.put_metric_data(
     *   Namespace='LangGraphAgent',
     *   MetricData=[{
     *     'MetricName': 'ToolCallSuccess',
     *     'Value': 1,
     *     'Unit': 'Count',
     *     'Dimensions': [{'Name': 'ToolName', 'Value': 'tavily_search'}]
     *   }]
     * )
     *
     * Then create alarm on that custom metric!
     */
  }
}
