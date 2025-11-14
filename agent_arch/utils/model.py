from litellm import acompletion
from agent_arch.utils.model_response import ErrorTracker, ModelResponse, Performance
from agent_arch.utils.perf_stats import PerfStats
import time
import random
import asyncio
from openai import AsyncOpenAI
import os
import json
import dotenv

dotenv.load_dotenv()


def get_model_info(model_name: str) -> dict:
    try:
        return json.loads(os.getenv(f"{model_name.upper()}_CONFIG"))
    except Exception as e:
        raise RuntimeError(f"❌ Error getting model info for {model_name}: {e}")


file_path = os.getenv("VERTEX_APPLICATION_CREDENTIALS")
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)
vertex_credentials_json = json.dumps(vertex_credentials)


class Model:
    def __init__(self):
        self.max_retries = 10
        self.base_delay = 2

    async def generate_text(self, model_config, messages, tools, inference_type):
        try:
            provider = model_config["provider"]
            model_name = model_config["name"]
            model_parameters = model_config["parameters"]

            call_start_time = time.time()
            if inference_type == "lite":
                chat_completion = await acompletion(
                    model=f"{provider}/{model_name}",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **({"allowed_openai_params": ["reasoning_effort"]} if "gpt-oss" in model_name else {}),
                    **({"allowed_openai_params": ["tools", "tool_choice"]} if "bedrock" in provider else {}),
                    **({"vertex_credentials": vertex_credentials_json} if provider == "vertex_ai" else {}),
                    **model_parameters
                )
            else:
                client = AsyncOpenAI(
                    base_url=model_config["endpoint"],
                    api_key=model_config["api_key"]
                )

                chat_completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **model_parameters
                )

            call_end_time = time.time()

            llm_response = chat_completion.choices[0].message.content or ""
            tool_calls = chat_completion.choices[0].message.tool_calls
            stop_reason = chat_completion.choices[0].finish_reason
            reasoning_response = getattr(
                chat_completion.choices[0].message, "reasoning_content", None
            )
            response_tokens = chat_completion.usage.completion_tokens
            reasoning_tokens = getattr(chat_completion.usage, "reasoning_tokens", None)
            latency = call_end_time - call_start_time
            prompt_tokens = chat_completion.usage.prompt_tokens
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
                raw_response=chat_completion.model_dump_json(),
                raw_response_object=chat_completion,
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
        except Exception as e:
            print(f"Actual exception: {e}")
            raise RetryableModelError(
                "Raising error for retry",
                {"type": "Error", "details": str(e)},
            )

    async def generate_text_with_retry(
            self,
            record_id,
            inference_type,
            model_config,
            messages,
            tools,
            model_name_with_call_id,
            tool_choice="auto",
    ):
        """
        Handles retry logic with exponential backoff for any model implementation.
        """

        perf_stats = PerfStats()
        error_tracker = ErrorTracker()
        start_time = time.time()

        attempt = 0
        while attempt <= self.max_retries:
            try:
                # Call the model-specific implementation
                model_response = await self.generate_text(
                    model_config,
                    messages,
                    tools,
                    inference_type
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

async def model_call(
        record_id,
        model_config,
        messages,
        tools,
        model_name_with_call_id,
        tool_choice="auto",
):
    client = Model()
    response = await client.generate_text_with_retry(
        record_id=record_id,
        inference_type=model_config["inference_type"],
        model_config=model_config,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        model_name_with_call_id=model_name_with_call_id,
    )
    return response

class RetryableModelError(Exception):
    """Used to signal errors that are safe to retry."""

    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}