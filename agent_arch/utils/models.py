import asyncio
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import dotenv
from aiobotocore.session import get_session
from google import genai
from google.auth import default
from google.auth.transport import requests
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig, Tool
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageToolCall
from agent_arch.utils.model_response import ErrorTracker, ModelResponse, Performance
from agent_arch.utils.perf_stats import PerfStats
from agent_arch.utils.util import remove_additional_properties

dotenv.load_dotenv()


def get_model_info(model_name: str) -> dict:
    try:
        return json.loads(os.getenv(f"{model_name.upper()}_CONFIG"))
    except Exception as e:
        raise RuntimeError(f"❌ Error getting model info for {model_name}: {e}")


async def model_call(
    record_id,
    model_config,
    messages,
    tools,
    model_name_with_call_id,
    tool_choice="auto",
):
    if "gemini" in model_config["name"]:
        client = GeminiModelClient(model_config=model_config)
    elif "claude" in model_config["name"]:
        client = BedrockClaudeClient(model_config=model_config)
    else:
        client = OpenAIModelClient(model_config=model_config)

    response = await client.generate_text_with_retry(
        record_id=record_id,
        model_config=model_config,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        model_name_with_call_id=model_name_with_call_id,
    )
    return response


class BaseModelClient:
    """Base class for model clients with retry functionality."""

    def __init__(self, model_config, max_retries=1, base_delay=1.0):
        self.model_config = model_config
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def generate_text_with_retry(
        self,
        record_id,
        model_config,
        messages,
        tools,
        model_name_with_call_id,
        tool_choice="auto",
    ):
        """
        Handles retry logic with exponential backoff for any model implementation.
        This method calls the model-specific generate_text method.
        """

        perf_stats = PerfStats()
        error_tracker = ErrorTracker()
        start_time = time.time()

        attempt = 0
        while attempt <= self.max_retries:
            try:
                # Call the model-specific implementation
                model_response = await self.generate_text(
                    record_id,
                    model_config,
                    messages,
                    tools,
                    model_name_with_call_id,
                    tool_choice,
                )

                perf_stats.add(
                    model_config["name"],
                    "id",
                    record_id + "_" + model_name_with_call_id,
                )
                perf_stats.add_all_stats(model_config["name"], model_response)

                return model_response

            except RetryableModelError as e:
                attempt += 1
                delay = self.base_delay * (2**attempt) + random.uniform(0, 0.5)
                print(
                    f"⚠️ Model call failed (attempt {attempt}/{self.max_retries}) — {type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(delay)
                else:
                    # LOG AS A SERVER FAILURE
                    print(f"❌ Exceeded maximum retry attempts ({self.max_retries}).")

        # If we've exceeded max retries, create error response and log stats
        end_time = time.time()
        final_response_code = 429 if error_tracker.rate_limit > 0 else 500

        model_response = ModelResponse(
            input_prompt=messages,
            llm_response="",
            is_failure=True,
            raw_response=f"Exceeded maximum retry attempts ({self.max_retries}). Last error tracked.",
            raw_response_object="",
            response_code=final_response_code,
            error_tracker=error_tracker,
            model_parameters=model_config,
            performance=None,
            wait_time=int((end_time - start_time) * 1000),
        )
        perf_stats.add(
            model_config["name"], "id", record_id + "_" + model_name_with_call_id
        )
        perf_stats.add_all_stats(model_name_with_call_id, model_response)

        return model_response

    async def generate_text(
        self,
        record_id,
        model_config,
        messages,
        tools,
        model_name_with_call_id,
        tool_choice="auto",
    ):
        """
        Abstract method to be overridden by specific model implementations.
        Should contain the actual API call logic without retry handling.
        Should never raise exceptions - always return a ModelResponse object.
        """
        raise NotImplementedError("Subclasses must implement generate_text method")


@dataclass
class Endpoint:
    """Represents a single OpenAI/Azure endpoint configuration"""

    endpoint: str
    api_key: str
    api_version: str = None  # Only needed for Azure
    provider: str = "openai"  # "openai" or "azure"
    rate_limit_reset_time: float = None
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    current_requests: int = 0  # Track concurrent requests
    max_concurrent_requests: int = 100  # Reasonable default


class EndpointManager:
    """Shared endpoint manager for better batch handling"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._endpoint_stats = defaultdict(
            lambda: {
                "total_requests": 0,
                "successful_requests": 0,
                "rate_limited_count": 0,
                "last_success_time": None,
            }
        )

    async def get_next_available_endpoint(
        self, endpoints: list[Endpoint], current_index: int = 0
    ) -> int:
        """Thread-safe endpoint selection with load balancing"""
        async with self._lock:
            current_time = time.time()
            available_endpoints = []

            # Find all available endpoints
            for i, endpoint in enumerate(endpoints):
                # Skip if rate limited and still in cooldown
                if (
                    endpoint.rate_limit_reset_time
                    and current_time < endpoint.rate_limit_reset_time
                ):
                    continue

                # Skip if too many consecutive failures
                if endpoint.consecutive_failures >= endpoint.max_consecutive_failures:
                    continue

                # Skip if too many concurrent requests
                if endpoint.current_requests >= endpoint.max_concurrent_requests:
                    continue

                available_endpoints.append(i)

            if not available_endpoints:
                return None

            # Load balance: prefer endpoints with fewer current requests
            available_endpoints.sort(
                key=lambda i: (
                    endpoints[i].current_requests,  # Fewer current requests first
                    endpoints[i].consecutive_failures,  # Fewer failures first
                    -self._endpoint_stats[i][
                        "successful_requests"
                    ],  # More successful requests first
                )
            )

            return available_endpoints[0]

    async def mark_request_start(self, endpoint_idx: int, endpoints: list[Endpoint]):
        """Mark that a request is starting on this endpoint"""
        async with self._lock:
            endpoints[endpoint_idx].current_requests += 1
            self._endpoint_stats[endpoint_idx]["total_requests"] += 1

    async def mark_request_end(
        self, endpoint_idx: int, endpoints: list[Endpoint], success: bool = True
    ):
        """Mark that a request has ended on this endpoint"""
        async with self._lock:
            endpoints[endpoint_idx].current_requests = max(
                0, endpoints[endpoint_idx].current_requests - 1
            )
            if success:
                endpoints[endpoint_idx].consecutive_failures = 0
                endpoints[endpoint_idx].rate_limit_reset_time = None
                self._endpoint_stats[endpoint_idx]["successful_requests"] += 1
                self._endpoint_stats[endpoint_idx]["last_success_time"] = time.time()

    async def mark_rate_limit(
        self, endpoint_idx: int, endpoints: list[Endpoint], reset_time: float = None
    ):
        """Mark an endpoint as rate limited"""
        async with self._lock:
            endpoint = endpoints[endpoint_idx]
            if reset_time:
                endpoint.rate_limit_reset_time = reset_time
            else:
                endpoint.rate_limit_reset_time = time.time() + 60

            self._endpoint_stats[endpoint_idx]["rate_limited_count"] += 1
            endpoint_name = self._get_endpoint_name(endpoint)
            provider_prefix = "🔵 Azure" if endpoint.provider == "azure" else "🟢 OpenAI"
            print(
                f"🚫 {provider_prefix} endpoint {endpoint_idx} ({endpoint_name}) rate limited until {endpoint.rate_limit_reset_time}"
            )

    async def mark_failure(self, endpoint_idx: int, endpoints: list[Endpoint]):
        """Mark an endpoint failure"""
        async with self._lock:
            endpoints[endpoint_idx].consecutive_failures += 1
            endpoint_name = self._get_endpoint_name(endpoints[endpoint_idx])
            provider_prefix = (
                "🔵 Azure" if endpoints[endpoint_idx].provider == "azure" else "🟢 OpenAI"
            )
            print(
                f"⚠️ {provider_prefix} endpoint {endpoint_idx} ({endpoint_name}) failure count: {endpoints[endpoint_idx].consecutive_failures}"
            )

    def _get_endpoint_name(self, endpoint: Endpoint) -> str:
        """Extract a readable name from endpoint URL"""
        if endpoint.provider == "azure":
            return endpoint.endpoint.split("//")[1].split(".")[0]
        else:
            # For OpenAI, just return "openai" or extract from custom base URLs
            if "api.openai.com" in endpoint.endpoint:
                return "openai"
            else:
                return endpoint.endpoint.split("//")[1].split(".")[0]

    def get_stats(self, endpoints: list[Endpoint]) -> dict:
        """Get current endpoint statistics"""
        stats = {}
        for i, endpoint in enumerate(endpoints):
            endpoint_name = self._get_endpoint_name(endpoint)
            stats[f"endpoint_{i}_{endpoint.provider}_{endpoint_name}"] = {
                "provider": endpoint.provider,
                "current_requests": endpoint.current_requests,
                "consecutive_failures": endpoint.consecutive_failures,
                "rate_limited": endpoint.rate_limit_reset_time is not None,
                **self._endpoint_stats[i],
            }
        return stats


_endpoint_manager = EndpointManager()


class OpenAIModelClient(BaseModelClient):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        # Parse endpoints from model_config (supports both OpenAI and Azure)
        self.endpoints = self._parse_endpoints(model_config)
        self.current_endpoint_index = 0
        self.clients = {}  # Cache clients for each endpoint
        self.endpoint_manager = _endpoint_manager  # Use global manager for shared state

        # Initialize clients for all endpoints
        self._initialize_clients()

    def _parse_endpoints(self, model_config) -> list[Endpoint]:
        """Parse endpoint configurations from model_config"""
        endpoints = []

        # Check if we have multiple endpoints configured
        if "endpoints" in model_config:
            for endpoint_config in model_config["endpoints"]:
                provider = self._detect_provider(endpoint_config)
                endpoints.append(
                    Endpoint(
                        endpoint=endpoint_config["endpoint"],
                        api_key=endpoint_config["api_key"],
                        api_version=endpoint_config.get(
                            "api_version"
                        ),  # Only for Azure
                        provider=provider,
                        max_concurrent_requests=endpoint_config.get(
                            "max_concurrent_requests", 100
                        ),
                    )
                )
        else:
            # Single endpoint format (backward compatibility)
            provider = self._detect_provider(model_config)
            endpoints.append(
                Endpoint(
                    endpoint=model_config["endpoint"],
                    api_key=model_config["api_key"],
                    api_version=model_config.get("api_version"),  # Only for Azure
                    provider=provider,
                    max_concurrent_requests=model_config.get(
                        "max_concurrent_requests", 100
                    ),
                )
            )

        return endpoints

    def _detect_provider(self, config) -> str:
        """Detect whether this is an Azure or OpenAI configuration"""
        # Check for explicit provider specification
        if "provider" in config:
            return config["provider"].lower()

        # Auto-detect based on configuration
        if "api_version" in config and config["api_version"]:
            return "azure"

        # Check endpoint URL patterns
        endpoint = config.get("endpoint", "")
        if "openai.azure.com" in endpoint or "cognitiveservices.azure.com" in endpoint:
            return "azure"
        elif "api.openai.com" in endpoint:
            return "openai"

        # Default to OpenAI if unclear
        return "openai"

    def _initialize_clients(self):
        """Initialize OpenAI/Azure clients for all endpoints"""
        for i, endpoint in enumerate(self.endpoints):
            if endpoint.provider == "azure":
                self.clients[i] = AsyncAzureOpenAI(
                    api_key=endpoint.api_key,
                    api_version=endpoint.api_version,
                    azure_endpoint=endpoint.endpoint,
                )
            else:
                self.clients[i] = AsyncOpenAI(
                    api_key=endpoint.api_key,
                    base_url=endpoint.endpoint
                    if endpoint.endpoint != "https://api.openai.com/v1"
                    else None,
                )

    async def generate_text(
        self,
        record_id,
        model_config,
        messages,
        tools,
        model_name_with_call_id,
        tool_choice="auto",
    ):
        """
        Single attempt at generating text with endpoint switching.
        Returns ModelResponse on success or raises RetryableModelError for retryable failures.
        """
        # For Azure, use deployment_id, for OpenAI use model name directly
        model_name = model_config["name"]
        tools_dict = {"tools": tools, "tool_choice": tool_choice} if tools else {}

        # Try each available endpoint once
        for _ in range(len(self.endpoints)):
            endpoint_idx = await self.endpoint_manager.get_next_available_endpoint(
                self.endpoints, self.current_endpoint_index
            )

            if endpoint_idx is None:
                # All endpoints busy/rate-limited - let base class retry later
                raise RetryableModelError(
                    "All endpoints busy/rate-limited", {"type": "AllEndpointsBusy"}
                )

            self.current_endpoint_index = endpoint_idx
            client = self.clients[endpoint_idx]
            endpoint = self.endpoints[endpoint_idx]

            # Mark request start
            await self.endpoint_manager.mark_request_start(endpoint_idx, self.endpoints)

            endpoint_name = self.endpoint_manager._get_endpoint_name(endpoint)

            try:
                call_start_time = time.time()
                response = await client.chat.completions.create(
                    model=model_name, messages=messages, **tools_dict
                )
                call_end_time = time.time()

                # Success! Mark request end
                await self.endpoint_manager.mark_request_end(
                    endpoint_idx, self.endpoints, success=True
                )

                # Process response
                llm_response = response.choices[0].message.content or ""
                tool_calls = response.choices[0].message.tool_calls
                stop_reason = response.choices[0].finish_reason

                latency = call_end_time - call_start_time
                prompt_tokens = response.usage.prompt_tokens
                reasoning_response = getattr(
                    response.choices[0].message, "reasoning_content", None
                )
                response_tokens = response.usage.completion_tokens
                reasoning_tokens = getattr(response.usage, "reasoning_tokens", None)
                time_per_token = (
                    latency / response_tokens if response_tokens > 0 else None
                )

                performance = Performance(
                    latency=latency,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    reasoning_tokens=reasoning_tokens,
                    time_per_token=time_per_token,
                    relative_output_tokens=response_tokens,
                )

                model_response = ModelResponse(
                    input_prompt=messages,
                    llm_response=llm_response,
                    reasoning_response=reasoning_response,
                    raw_response=response.model_dump_json(),
                    raw_response_object=response,
                    response_code=200,
                    stop_reason=stop_reason,
                    tool_calls=[tool_call.model_dump() for tool_call in tool_calls]
                    if tool_calls
                    else None,
                    performance=performance,
                    wait_time=int((call_end_time - call_start_time) * 1000),
                    error_tracker=None,
                    model_parameters=model_config,
                    model_info=model_config,
                )

                return model_response

            except RateLimitError as e:
                await self.endpoint_manager.mark_request_end(
                    endpoint_idx, self.endpoints, success=False
                )
                provider_prefix = (
                    "🔵 Azure" if endpoint.provider == "azure" else "🟢 OpenAI"
                )
                print(
                    f"🚫 Rate limit hit on {provider_prefix} endpoint {endpoint_idx} ({endpoint_name})"
                )

                # Try to extract reset time from headers
                reset_time = None
                if (
                    hasattr(e, "response")
                    and e.response
                    and hasattr(e.response, "headers")
                ):
                    # Azure headers
                    reset_requests = e.response.headers.get(
                        "x-ratelimit-reset-requests"
                    )
                    reset_tokens = e.response.headers.get("x-ratelimit-reset-tokens")
                    # OpenAI headers
                    retry_after = e.response.headers.get("retry-after")

                    if reset_requests:
                        try:
                            reset_time = time.time() + float(reset_requests)
                        except ValueError:
                            pass
                    elif reset_tokens:
                        try:
                            reset_time = time.time() + float(reset_tokens)
                        except ValueError:
                            pass
                    elif retry_after:
                        try:
                            reset_time = time.time() + float(retry_after)
                        except ValueError:
                            pass

                await self.endpoint_manager.mark_rate_limit(
                    endpoint_idx, self.endpoints, reset_time
                )
                # Try next endpoint if available
                continue

            except (APIStatusError, APITimeoutError, APIError, APIConnectionError) as e:
                await self.endpoint_manager.mark_request_end(
                    endpoint_idx, self.endpoints, success=False
                )
                await self.endpoint_manager.mark_failure(endpoint_idx, self.endpoints)
                provider_prefix = (
                    "🔵 Azure" if endpoint.provider == "azure" else "🟢 OpenAI"
                )
                print(
                    f"⚠️ {provider_prefix} API error on endpoint {endpoint_idx} ({endpoint_name}): {type(e).__name__}: {e}"
                )
                # Try next endpoint if available
                continue

            except Exception as e:
                await self.endpoint_manager.mark_request_end(
                    endpoint_idx, self.endpoints, success=False
                )
                await self.endpoint_manager.mark_failure(endpoint_idx, self.endpoints)
                provider_prefix = (
                    "🔵 Azure" if endpoint.provider == "azure" else "🟢 OpenAI"
                )
                print(
                    f"❌ Unexpected error on {provider_prefix} endpoint {endpoint_idx} ({endpoint_name}): {str(e)}"
                )

                # For unexpected errors, return error response instead of raising
                response_text = {"code": 500, "message": str(e)}
                return ModelResponse(
                    input_prompt=messages,
                    model_parameters=model_config,
                    llm_response=json.dumps(response_text),
                    raw_response=json.dumps(response_text),
                    response_code=500,
                    performance=None,
                    wait_time=0,
                    is_failure=True,
                )

        # If we've tried all endpoints and none worked, raise retryable error
        raise RetryableModelError(
            "All endpoints failed in this attempt", {"type": "AllEndpointsFailed"}
        )

    def get_endpoint_stats(self) -> dict:
        """Get current endpoint statistics for monitoring"""
        return self.endpoint_manager.get_stats(self.endpoints)


class BedrockClaudeClient(BaseModelClient):
    """BedrockClaude-specific implementation."""

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.model_config = model_config
        self.reasoning_config = (
            {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": model_config.get("thinking_budget_tokens", 4000),
                }
            }
            if model_config.get("thinking", False)
            else {}
        )
        self.model_endpoint = model_config["endpoint"]

    @staticmethod
    def _get_role(role):
        """Convert "tool" role to "user" role for bedrock."""
        return "user" if (role == "tool" or role == "system") else role

    @staticmethod
    def apply_formatter(messages: list[dict]) -> list[dict]:
        """Apply formatting to convert chat completion messages to bedrock format."""
        formatted_messages = []
        sys_content = ""
        for message in messages:
            original_role = message.get("role", "")
            role = BedrockClaudeClient._get_role(message.get("role"))
            content = message.get("content")
            if original_role == "system":
                # move system content to following turn
                sys_content = content + "\n\n"
            if isinstance(content, list):
                content_to_append = []
                for msg in content:
                    if isinstance(msg, ChatCompletionMessageToolCall):
                        content_to_append.append(
                            {
                                "toolUse": {
                                    "toolUseId": msg.id,
                                    "name": msg.function.name,
                                    "input": json.loads(msg.function.arguments),
                                }
                            }
                        )
                    elif isinstance(msg, dict) and "toolUse" in msg:
                        content_to_append.append(msg)
                    elif message.get("type", "") == "text":
                        content_to_append.append(
                            {"text": sys_content + message["text"]}
                        )
                        sys_content = ""
                    # elif msg.get("type", "") == "text":
                    #     content_to_append.append({"text": msg["text"]})
                    else:
                        raise NotImplementedError(
                            f"Content Type '{msg['type']}' is not supported."
                        )
                formatted_messages.append({"role": role, "content": content_to_append})

            elif isinstance(content, dict):
                if "tool_call_id" in content:
                    formatted_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        "toolUseId": content["tool_call_id"],
                                        "content": [{"text": content["content"]}],
                                    }
                                }
                            ],
                        }
                    )
                else:
                    formatted_messages.append({"role": role, "content": [content]})
            elif message.get("role") == "tool":
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message["tool_call_id"],
                                    "content": [{"text": message["content"]}],
                                }
                            }
                        ],
                    }
                )

            elif isinstance(content, str):
                formatted_messages.append(
                    {"role": role, "content": [{"text": content}]}
                )

        return formatted_messages

    async def generate_text(
        self,
        record_id,
        model_config,
        messages,
        tools,
        model_name_with_call_id,
        tool_choice="auto",
    ):
        """
        BedrockClaude-specific implementation without retry logic.
        Never raises exceptions - always returns a ModelResponse object.
        """

        formatted_messages = BedrockClaudeClient.apply_formatter(messages)
        tool_config = BedrockClaudeClient.format_tool_config(tools)
        mapped_params = model_config["parameters"]
        reasoning_config = {}
        max_tokens = mapped_params.get("maxTokens", 10000)
        if (
            self.reasoning_config and max_tokens >= 1024
        ):  # ignore reasoning config for test call
            reasoning_config = self.reasoning_config
            if self.reasoning_config["thinking"]["budget_tokens"] >= max_tokens:
                # `max_tokens` must be greater than `thinking.budget_tokens`
                reasoning_config["thinking"]["budget_tokens"] = (
                    mapped_params["maxTokens"] / 2
                )
            if (
                mapped_params.get("temperature")
                and mapped_params.get("temperature") < 1
            ):
                # `temperature` may only be set to 1 when thinking is enabled.
                mapped_params["temperature"] = 1

        wait_time = None
        session = get_session()
        async with session.create_client(
            "bedrock-runtime", region_name="us-east-1"
        ) as self.client:
            start_time = time.time()
            try:
                response = await self.client.converse(
                    modelId=self.model_endpoint,
                    messages=formatted_messages,
                    inferenceConfig=mapped_params,
                    **({} if tool_config == {} else {"toolConfig": tool_config}),
                    additionalModelRequestFields=reasoning_config,
                )
                elapsed_time = time.time() - start_time

                llm_response = " "
                reasoning_response = None
                tool_calls = None
                model_resp = response["output"]["message"]["content"]
                for resp in model_resp:
                    if "text" in resp:
                        llm_response = resp["text"]
                    if "reasoningContent" in resp:
                        reasoning_response = resp["reasoningContent"]["reasoningText"][
                            "text"
                        ]
                    if "toolUse" in resp:
                        function_call = resp["toolUse"]

                        openai_tool_call = ChatCompletionMessageToolCall(
                            function={
                                "arguments": json.dumps(function_call["input"]),
                                "name": function_call["name"],
                            },
                            type="function",
                            id=function_call["toolUseId"],
                        )
                        tool_calls = [openai_tool_call]

                raw_response = (
                    llm_response
                    if not reasoning_response
                    else "<thinking>\n"
                    + reasoning_response
                    + "\n</thinking>\n"
                    + llm_response
                )

                stop_reason = response["stopReason"]
                return ModelResponse(
                    llm_response=llm_response,
                    raw_response=raw_response,
                    reasoning_response=reasoning_response,
                    raw_response_object=response,
                    model_parameters=mapped_params,
                    input_prompt=formatted_messages,
                    response_code=200,
                    stop_reason=stop_reason,
                    tool_calls=[tool_call.model_dump() for tool_call in tool_calls]
                    if tool_calls
                    else None,
                    wait_time=wait_time,
                    performance=Performance(
                        prompt_tokens=response["usage"]["inputTokens"]
                        if response.get("usage")
                        else 0,
                        latency=elapsed_time,
                        response_tokens=response["usage"]["outputTokens"]
                        if response.get("usage")
                        else 0,
                        reasoning_tokens=-1,
                        time_per_token=-1,
                        relative_output_tokens=-1,
                    ),
                )
            except (ClientError, Exception) as e:
                # Handle client error
                raise RetryableModelError(
                    "ClientError or Exception occurred",
                    {"type": "ClientError", "details": str(e)},
                )

            except Exception as e:
                print(f"Bedrock error not being retried: {e}")
                response_text = {
                    "response": "error",
                    "message": str(e),
                }
                return ModelResponse(
                    input_prompt=formatted_messages,
                    model_parameters={},
                    llm_response=json.dumps(response_text),
                    raw_response=json.dumps(response_text),
                    response_code=500,
                    performance=None,
                    wait_time=wait_time,
                )

    @staticmethod
    def format_tool_config(tools: list[dict]) -> dict:
        """Convert generic dict to bedrock tool dict."""
        seen_tool_names = set()
        formatted_tools = []
        for tool in tools:
            tool_name = tool["function"]["name"]
            if tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            formatted_tools.append(
                {
                    "toolSpec": {
                        "name": tool["function"]["name"],
                        "inputSchema": {"json": tool["function"]["parameters"]},
                    }
                }
            )
        return {"tools": formatted_tools}


class GeminiModelClient(BaseModelClient):
    """Gemini-specific implementation (example of how to extend for other models)."""

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.project = model_config["project"]
        self.location = model_config["location"]
        self.model_name = model_config["name"]
        self.client = None
        self.thinking_config = None
        self.model_config = model_config
        self.thinking_config = {}
        if model_config.get("thinking_model"):
            self.thinking_config = ThinkingConfig(**model_config.get("thinking_config"))

    @staticmethod
    def _get_role(role):
        """Convert "assistant" role to "model" role for gemini."""
        return "model" if role == "assistant" else role

    @staticmethod
    def apply_formatter(messages):
        formatted_messages = []

        for message in messages:
            # if already in gemini format just append
            if (
                isinstance(message, dict)
                and message.get("role") in {"user", "model"}
                and isinstance(message.get("parts"), list)
            ):
                formatted_messages.append(message)
                continue

            # if in chat completions format then conver to gemini
            if "role" in message and "content" in message:
                role = message["role"]
                content = message["content"]

                if role == "tool":
                    # convert to Gemini function response format tool becomes user
                    formatted_messages.append(
                        {
                            "role": "user",
                            "parts": [
                                Part.from_function_response(
                                    name=message["name"],
                                    response={"result": message["content"]},
                                )
                            ],
                        }
                    )

                elif (
                    role == "assistant"
                    and isinstance(content, list)
                    and isinstance(content[0], ChatCompletionMessageToolCall)
                ):
                    # assistant becomes model
                    formatted_messages.append(
                        {
                            "role": "model",
                            "parts": [
                                Part.from_function_call(
                                    name=content[0].function.name,
                                    args=json.loads(content[0].function.arguments),
                                )
                            ],
                        }
                    )

                else:
                    # rest is the same
                    mapped_role = GeminiModelClient._get_role(role)
                    formatted_messages.append(
                        {"role": mapped_role, "parts": [{"text": content}]}
                    )

            else:
                raise ValueError(f"Unrecognized message format: {message}")

        return formatted_messages

    @staticmethod
    def format_tool_config(passed_tools: list[dict]):
        """Convert generic dict to gemini tools."""
        tool_list = []
        seen_tool_names = set()
        for tool in passed_tools:
            tool_name = tool["function"]["name"]
            if tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            tool_function = tool["function"]
            # Recursively remove all additionalProperties
            remove_additional_properties(tool_function)
            # Remove "eval_type" from parameter properties
            if (
                "parameters" in tool_function
                and "properties" in tool_function["parameters"]
            ):
                for prop in tool_function["parameters"]["properties"].values():
                    if isinstance(prop, dict) and "eval_type" in prop:
                        prop.pop("eval_type")
            if tool_function.get("strict") is True:
                tool_function.pop("strict")
            tool_list.append(tool_function)

        tools = [Tool(function_declarations=tool_list)] if tool_list else []
        return tools

    async def generate_text(
        self,
        record_id,
        model_config,
        messages,
        passed_tools,
        model_name_with_call_id,
        tool_choice="auto",
    ):
        """
        Gemini-specific implementation without retry logic.
        Never raises exceptions - always returns a ModelResponse object.
        """
        formatted_messages = GeminiModelClient.apply_formatter(messages)

        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location
        )
        wait_time = None
        try:
            start_time = time.time()
            if (
                isinstance(formatted_messages, list)
                and formatted_messages[0].get("role") == "system"
            ):
                model_config["system_instruction"] = [
                    formatted_messages[0]["parts"][0]["text"]
                ]
                formatted_messages = formatted_messages[1:]
            tool_list = []

            for tool in passed_tools:
                tool_function = tool["function"]

                # Recursively remove all additionalProperties
                remove_additional_properties(tool_function)

                # Remove "eval_type" from parameter properties
                if (
                    "parameters" in tool_function
                    and "properties" in tool_function["parameters"]
                ):
                    for prop in tool_function["parameters"]["properties"].values():
                        if isinstance(prop, dict) and "eval_type" in prop:
                            prop.pop("eval_type")

                # Remove "strict": True if present
                if tool_function.get("strict") is True:
                    tool_function.pop("strict")

                tool_list.append(tool_function)

            mapped_params = model_config["parameters"]
            if self.thinking_config:
                mapped_params["thinking_config"] = self.thinking_config
            config = GenerateContentConfig(
                **mapped_params,
                tools=GeminiModelClient.format_tool_config(passed_tools),
            )
            response = await self.client.aio.models.generate_content(
                contents=formatted_messages, model=self.model_name, config=config
            )

            elapsed_time = time.time() - start_time
            reasoning_response = ""
            llm_response = ""
            tool_calls = None
            parts = response.candidates[0].content.parts
            if not parts:
                print("Response from Gemini is empty.")
                raise Exception("Response from Gemini is empty.")
            for part in parts:
                if part.thought:
                    reasoning_response += part.text
                elif part.function_call:
                    function_call = part.function_call
                    openai_tool_call = ChatCompletionMessageToolCall(
                        function={
                            "arguments": json.dumps(function_call.args),
                            "name": function_call.name,
                        },
                        type="function",
                        id="test",
                    )
                    tool_calls = [openai_tool_call]

                else:
                    llm_response += part.text
            if "response_logprobs" in model_config:
                response_out = {}
                response_out["text"] = response.text
                log_probs_content = response.candidates[0].logprobs.content[0].__dict__
                log_probs_content["top_logprobs"] = [
                    x.__dict__ for x in log_probs_content["top_logprobs"]
                ]
                response_out["logprobs"] = log_probs_content
                llm_response = json.dumps(response_out)
            stop_reason = response.candidates[0].finish_reason
            if response.usage_metadata.thoughts_token_count:
                reasoning_tokens = float(response.usage_metadata.thoughts_token_count)
            else:
                reasoning_tokens = None

            return ModelResponse(
                llm_response=llm_response,
                model_parameters=model_config,
                reasoning_response=reasoning_response,
                raw_response=llm_response,
                raw_response_object=response,
                input_prompt=formatted_messages,
                response_code=200,
                stop_reason=stop_reason,
                tool_calls=[tool_call.model_dump() for tool_call in tool_calls]
                if tool_calls
                else None,
                wait_time=wait_time,
                performance=Performance(
                    prompt_tokens=float(
                        response.usage_metadata.prompt_token_count
                        if response.usage_metadata.prompt_token_count
                        else 0
                    ),
                    latency=float(elapsed_time),
                    response_tokens=float(
                        response.usage_metadata.total_token_count
                        if response.usage_metadata.total_token_count
                        else 0
                    ),
                    reasoning_tokens=reasoning_tokens,
                    time_per_token=-1,
                    relative_output_tokens=-1,
                ),
            )

        except ClientError as e:
            raise RetryableModelError(
                "ClientError (RateLimit or Transient)",
                {"type": "ClientError", "details": str(e)},
            )

        except ServerError as e:
            raise RetryableModelError(
                "ServerError (Transient failure)",
                {"type": "ServerError", "details": str(e)},
            )

        except Exception as e:
            # Handle specific known issues that are safe to retry
            if "Max retries exceeded" in str(e):
                raise RetryableModelError(
                    "MaxRetryError", {"type": "MaxRetryError", "details": str(e)}
                )
            if "Response from Gemini is empty" in str(e):
                raise RetryableModelError(
                    "Empty Gemini response",
                    {"type": "EmptyResponse", "details": str(e)},
                )
            if not str(e):
                raise RetryableModelError(
                    "Unknown connect error", {"type": "UnknownError", "details": str(e)}
                )

            print("Gemini error not being retried: " + str(e))

            # If not retryable, wrap a safe error response
            response_text = {
                "response": "error",
                "message": str(e),
            }
            return ModelResponse(
                input_prompt=formatted_messages,
                model_parameters={},
                llm_response=json.dumps(response_text),
                raw_response=json.dumps(response_text),
                response_code=500,
                performance=None,
                wait_time=wait_time,
            )


class RetryableModelError(Exception):
    """Used to signal errors that are safe to retry."""

    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}
