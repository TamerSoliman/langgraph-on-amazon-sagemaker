"""
Agent Monitoring Implementation

Provides instrumentation for LangGraph agents running on AWS.

For AI/ML Scientists:
This is like instrumenting your training loop to track loss, accuracy, etc.
But for production inference, we track latency, errors, costs.
"""

import time
import boto3
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime
from contextlib import contextmanager


# =============================================================================
# AGENT MONITOR
# =============================================================================

class AgentMonitor:
    """
    Main monitoring class for LangGraph agents.

    For AI/ML Scientists:
    This is like a profiler + logger combined. It tracks:
    - Function call counts (like forward passes)
    - Execution time (like training time)
    - Errors (like NaN losses)
    - Custom metrics (like validation accuracy)

    Usage:
        monitor = AgentMonitor(namespace="MyApp", agent_name="my-agent")

        @monitor.track_invocation
        def run_agent(input):
            # Your agent code
            return output
    """

    def __init__(
        self,
        namespace: str = "LangGraphAgents",
        agent_name: str = "default",
        region: str = "us-east-1"
    ):
        """
        Initialize monitor.

        Args:
            namespace: CloudWatch namespace for metrics
            agent_name: Name of this agent (used in dimensions)
            region: AWS region
        """
        self.namespace = namespace
        self.agent_name = agent_name
        self.region = region

        # CloudWatch clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.logs = boto3.client('logs', region_name=region)

        # Logger
        self.logger = logging.getLogger(f"{namespace}.{agent_name}")

    # =========================================================================
    # DECORATORS
    # =========================================================================

    def track_invocation(self, func: Callable) -> Callable:
        """
        Decorator to track agent invocations.

        Tracks:
        - Total invocations
        - Success/failure
        - Latency
        - Errors

        For AI/ML Scientists:
        This is like wrapping your model's forward() method to track
        inference count, latency, and errors.

        Usage:
            @monitor.track_invocation
            def run_agent(question):
                return answer
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_type = None

            try:
                # Execute function
                result = func(*args, **kwargs)
                success = True
                return result

            except Exception as e:
                # Record error
                error_type = type(e).__name__
                self.record_error(error_type, str(e))
                raise

            finally:
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000

                self.put_metric("Invocations", 1, unit="Count")
                self.put_metric("Latency", latency_ms, unit="Milliseconds")

                if success:
                    self.put_metric("Success", 1, unit="Count")
                else:
                    self.put_metric("Errors", 1, unit="Count")
                    if error_type:
                        self.put_metric(
                            f"Error_{error_type}",
                            1,
                            unit="Count"
                        )

                # Log
                self.logger.info(
                    f"Invocation completed: success={success}, "
                    f"latency={latency_ms:.0f}ms, error={error_type}"
                )

        return wrapper

    def track_llm_call(self, func: Callable) -> Callable:
        """
        Decorator to track LLM calls.

        Tracks:
        - Number of LLM calls
        - LLM latency
        - Token usage (if available)

        For AI/ML Scientists:
        This tracks your model inference calls specifically, separate from
        other operations.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record metrics
                latency_ms = (time.time() - start_time) * 1000

                self.put_metric("LLMCalls", 1, unit="Count")
                self.put_metric("LLMLatency", latency_ms, unit="Milliseconds")

                # Try to extract token usage if available
                if hasattr(result, 'usage_metadata'):
                    input_tokens = result.usage_metadata.get('input_tokens', 0)
                    output_tokens = result.usage_metadata.get('output_tokens', 0)

                    self.put_metric("InputTokens", input_tokens, unit="Count")
                    self.put_metric("OutputTokens", output_tokens, unit="Count")

                return result

            except Exception as e:
                self.put_metric("LLMErrors", 1, unit="Count")
                raise

        return wrapper

    def track_tool_call(self, tool_name: str) -> Callable:
        """
        Decorator to track tool usage.

        Args:
            tool_name: Name of the tool being tracked

        For AI/ML Scientists:
        This tracks specific components of your system, like tracking
        individual layers or modules in your model.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False

                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result

                except Exception as e:
                    self.put_metric(
                        f"Tool_{tool_name}_Errors",
                        1,
                        unit="Count"
                    )
                    raise

                finally:
                    latency_ms = (time.time() - start_time) * 1000

                    self.put_metric(
                        f"Tool_{tool_name}_Calls",
                        1,
                        unit="Count"
                    )

                    self.put_metric(
                        f"Tool_{tool_name}_Latency",
                        latency_ms,
                        unit="Milliseconds"
                    )

                    if success:
                        self.put_metric(
                            f"Tool_{tool_name}_Success",
                            1,
                            unit="Count"
                        )

            return wrapper
        return decorator

    # =========================================================================
    # CONTEXT MANAGERS
    # =========================================================================

    @contextmanager
    def timer(self, operation_name: str):
        """
        Context manager for timing operations.

        For AI/ML Scientists:
        Like using `with torch.no_grad():` but for timing instead of
        disabling gradients.

        Usage:
            with monitor.timer("data_loading"):
                data = load_data()
        """

        start_time = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000

            self.put_metric(
                f"Operation_{operation_name}_Latency",
                latency_ms,
                unit="Milliseconds"
            )

    # =========================================================================
    # METRIC RECORDING
    # =========================================================================

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Send metric to CloudWatch.

        For AI/ML Scientists:
        Like logging a metric to Weights & Biases or TensorBoard.

        Args:
            metric_name: Name of metric
            value: Metric value
            unit: CloudWatch unit (Count, Milliseconds, etc.)
            dimensions: Additional dimensions for filtering
        """

        # Build dimensions
        metric_dimensions = [
            {"Name": "AgentName", "Value": self.agent_name}
        ]

        if dimensions:
            for key, val in dimensions.items():
                metric_dimensions.append({"Name": key, "Value": val})

        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Timestamp": datetime.utcnow(),
                        "Dimensions": metric_dimensions
                    }
                ]
            )

        except Exception as e:
            self.logger.warning(f"Failed to put metric {metric_name}: {e}")

    def record_metric(self, metric_name: str, value: float):
        """
        Convenience method for recording custom metrics.

        For AI/ML Scientists:
        Use this for domain-specific metrics like "retrieval_relevance"
        or "answer_quality".
        """

        self.put_metric(metric_name, value, unit="None")

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record error with context.

        For AI/ML Scientists:
        Like logging exceptions with full traceback and context for debugging.
        """

        # Put error metric
        self.put_metric(f"Error_{error_type}", 1, unit="Count")

        # Log error with context
        self.logger.error(
            f"Error occurred: {error_type} - {error_message}",
            extra=context or {}
        )

    # =========================================================================
    # COST TRACKING
    # =========================================================================

    def estimate_cost(
        self,
        llm_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        compute_seconds: float = 0,
        instance_type: str = "ml.g5.xlarge"
    ) -> float:
        """
        Estimate cost of invocation.

        For AI/ML Scientists:
        This is cost accounting for inference - like tracking GPU hours
        for training.

        Args:
            llm_calls: Number of LLM calls
            input_tokens: Total input tokens
            output_tokens: Total output tokens
            compute_seconds: Execution time
            instance_type: SageMaker instance type

        Returns:
            Estimated cost in USD
        """

        # SageMaker pricing (example: ml.g5.xlarge)
        instance_costs = {
            "ml.g5.xlarge": 1.006 / 3600,  # $1.006/hour → per second
            "ml.g5.2xlarge": 1.515 / 3600,
            "ml.g5.4xlarge": 2.534 / 3600,
        }

        compute_cost_per_sec = instance_costs.get(instance_type, 1.0 / 3600)

        # Estimate costs
        compute_cost = compute_seconds * compute_cost_per_sec

        # Token costs (if using hosted LLM like Bedrock)
        # Example: $0.0008 per 1K input tokens, $0.0024 per 1K output tokens
        # For SageMaker, token costs are included in compute
        token_cost = 0.0

        total_cost = compute_cost + token_cost

        # Record metric
        self.put_metric("EstimatedCost", total_cost * 1000, unit="None")  # millidollars

        return total_cost

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_group: str = "/aws/langgraph/agents",
    include_trace_id: bool = True
) -> logging.Logger:
    """
    Configure structured logging with CloudWatch integration.

    For AI/ML Scientists:
    This sets up logging similar to how you'd configure logging for
    training runs - structured, searchable, with context.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_group: CloudWatch log group name
        include_trace_id: Include trace ID in logs (for correlation)

    Returns:
        Configured logger
    """

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with structured format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Note: CloudWatch Logs handler would be added here
    # For Lambda, logs automatically go to CloudWatch

    return logger


# =============================================================================
# ALARMS
# =============================================================================

def create_alarms(
    agent_name: str,
    namespace: str = "LangGraphAgents",
    sns_topic_arn: str = None,
    error_rate_threshold: float = 0.05,
    latency_p99_threshold: float = 10000,
    region: str = "us-east-1"
) -> list:
    """
    Create CloudWatch alarms for agent monitoring.

    For AI/ML Scientists:
    This is like setting up alerts for training - if loss spikes or
    accuracy drops, get notified.

    Args:
        agent_name: Name of agent to monitor
        namespace: CloudWatch namespace
        sns_topic_arn: SNS topic for notifications
        error_rate_threshold: Alarm if error rate exceeds this (0.05 = 5%)
        latency_p99_threshold: Alarm if p99 latency exceeds this (ms)
        region: AWS region

    Returns:
        List of created alarm ARNs
    """

    cloudwatch = boto3.client('cloudwatch', region_name=region)
    alarms = []

    # High error rate alarm
    alarm_name = f"{agent_name}-high-error-rate"

    try:
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='Errors',
            Namespace=namespace,
            Period=300,  # 5 minutes
            Statistic='Sum',
            Threshold=error_rate_threshold * 100,  # Convert to count
            ActionsEnabled=True,
            AlarmActions=[sns_topic_arn] if sns_topic_arn else [],
            AlarmDescription=f'Alarm when {agent_name} error rate exceeds {error_rate_threshold*100}%',
            Dimensions=[
                {'Name': 'AgentName', 'Value': agent_name}
            ]
        )

        alarms.append(alarm_name)

    except Exception as e:
        logging.warning(f"Failed to create alarm {alarm_name}: {e}")

    # High latency alarm
    alarm_name = f"{agent_name}-high-latency"

    try:
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='Latency',
            Namespace=namespace,
            Period=300,
            Statistic='p99',  # 99th percentile
            Threshold=latency_p99_threshold,
            ActionsEnabled=True,
            AlarmActions=[sns_topic_arn] if sns_topic_arn else [],
            AlarmDescription=f'Alarm when {agent_name} p99 latency exceeds {latency_p99_threshold}ms',
            Dimensions=[
                {'Name': 'AgentName', 'Value': agent_name}
            ]
        )

        alarms.append(alarm_name)

    except Exception as e:
        logging.warning(f"Failed to create alarm {alarm_name}: {e}")

    return alarms


# =============================================================================
# X-RAY TRACING
# =============================================================================

def enable_xray_tracing():
    """
    Enable AWS X-Ray distributed tracing.

    For AI/ML Scientists:
    X-Ray is like a profiler that shows you exactly where time is spent
    in each request, across all services.
    """

    try:
        from aws_xray_sdk.core import xray_recorder, patch_all

        # Patch all supported libraries
        patch_all()

        print("✓ X-Ray tracing enabled")

    except ImportError:
        print("⚠️  aws-xray-sdk not installed. Run: pip install aws-xray-sdk")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the monitoring system.

    Usage:
        python agent_monitor.py
    """

    print("="*70)
    print("Agent Monitor Test")
    print("="*70)

    # Create monitor
    monitor = AgentMonitor(
        namespace="TestNamespace",
        agent_name="test-agent"
    )

    # Test invocation tracking
    @monitor.track_invocation
    def test_function():
        time.sleep(0.1)  # Simulate work
        return "success"

    print("\n1. Testing invocation tracking...")
    result = test_function()
    print(f"   Result: {result}")

    # Test timer
    print("\n2. Testing timer...")
    with monitor.timer("test_operation"):
        time.sleep(0.05)
    print("   Timer completed")

    # Test custom metrics
    print("\n3. Testing custom metrics...")
    monitor.record_metric("test_metric", 42.0)
    print("   Metric recorded")

    # Test error tracking
    print("\n4. Testing error tracking...")
    try:
        raise ValueError("Test error")
    except ValueError as e:
        monitor.record_error("ValueError", str(e), context={"test": True})
    print("   Error tracked")

    print("\n" + "="*70)
    print("All tests completed!")
    print("Check CloudWatch console for metrics:")
    print(f"  Namespace: TestNamespace")
    print(f"  Agent: test-agent")
    print("="*70)
