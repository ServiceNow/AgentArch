import json
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.model_response import ErrorTracker, ModelResponse


class PerfStats:
    """Singleton class to track performance statistics across entire CLAE run."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.stats = defaultdict(list)

    def add(self, model, metric, value):
        self.stats[model].append({metric: value})

    def clear(self):
        self.stats.clear()

    def rearranged_data(self):
        rearranged_data = defaultdict(lambda: defaultdict(list))
        for model, entries in self.stats.items():
            for entry in entries:
                for metric, value in entry.items():
                    rearranged_data[model][metric].append(value)
        return rearranged_data

    def compute_statistics(self, data):
        statistics = defaultdict(dict)
        for model, metrics in data.items():
            for metric, values in metrics.items():
                if metric == "raw_response_object":
                    continue
                if metric == "response_code":
                    statistics[model][metric] = {
                        str(code): values.count(code) for code in set(values)
                    }
                    continue
                if isinstance(values, list):
                    if all(isinstance(item, list) for item in values):
                        print("Selecting last value in list for computing perf stats")
                        values = values[-1]
                    elif all(isinstance(item, (dict, str)) for item in values):
                        continue
                try:
                    np_values = np.array(values)
                except Exception as e:
                    print(f"Error converting {values} for {metric} ({model}): {e}")
                    continue
                np_values = np_values[np_values != np.array(None)]
                if len(np_values) == 0:
                    statistics[model][metric] = {"average": None}
                elif isinstance(values[0], bool):
                    statistics[model][metric] = {
                        "average": float(np.mean(np_values)),
                        "count": float(np.sum(np_values)),
                    }
                elif isinstance(values[0], float):
                    statistics[model][metric] = {
                        "min": float(np.nanmin(np_values)),
                        "max": float(np.nanmax(np_values)),
                        "average": float(np.nanmean(np_values)),
                        "quantiles": {
                            "25%": float(np.nanpercentile(np_values, 25)),
                            "50%": float(np.nanpercentile(np_values, 50)),
                            "75%": float(np.nanpercentile(np_values, 75)),
                            "95%": float(np.nanpercentile(np_values, 95)),
                            "99%": float(np.nanpercentile(np_values, 99)),
                        },
                    }
                elif isinstance(values[0], ErrorTracker):
                    statistics[model][metric] = {
                        "rate_limit (429)": values[0].rate_limit,
                        "connection_error (599)": values[0].connection_error,
                        "api_error (401)": values[0].api_error,
                        "request_timeout (408)": values[0].request_timeout,
                        "internal_server (500)": values[0].internal_server,
                        "other": values[0].other,
                    }
        return statistics

    def statistics(self):
        if not self.stats:
            return {}
        return self.compute_statistics(self.rearranged_data())

    def record_level_data(self):
        return self._generate_record_level_view(self.rearranged_data())

    def _generate_record_level_view(self, data) -> dict[str, pd.DataFrame]:
        result = {}
        for model_name, metrics in data.items():
            try:
                result[model_name] = pd.DataFrame(metrics)
            except ValueError as e:
                # Handle length mismatch by padding shorter arrays
                if "All arrays must be of the same length" in str(e):
                    max_length = max(len(values) if isinstance(values, list) else 1
                                     for values in metrics.values())

                    padded_metrics = {}
                    for metric_name, values in metrics.items():
                        if isinstance(values, list):
                            if len(values) < max_length:
                                # Pad with appropriate default values
                                if metric_name in ['is_failure', 'is_empty', 'success']:
                                    default_value = False
                                elif metric_name in ['input_tokens', 'output_tokens', 'reasoning_tokens', 'time_per_token']:
                                    default_value = 0.0
                                elif metric_name == 'response_code':
                                    default_value = 200
                                else:
                                    default_value = None

                                padded_values = values + [default_value] * (max_length - len(values))
                                padded_metrics[metric_name] = padded_values
                            else:
                                padded_metrics[metric_name] = values
                        else:
                            # Single value, convert to list
                            padded_metrics[metric_name] = [values] * max_length

                    result[model_name] = pd.DataFrame(padded_metrics)
                else:
                    raise e

        return result

    def generate_summary(self, filename):
        record_level_data = self.record_level_data()
        for model_name, df in record_level_data.items():
            df.to_csv(f"{filename}_{model_name}_record_level.csv", index=False)
        if self.statistics():
            with open(f"{filename}.json", "w") as f:
                json.dump(self.statistics(), f, indent=4)

    def add_all_stats(self, model_name: str, model_response: ModelResponse):
        llm_response = model_response.llm_response if 200 <= model_response.response_code < 300 else ""
        self.add(
            model_name,
            "input",
            str(model_response.input_prompt)
            if isinstance(model_response.input_prompt, list)
            else model_response.input_prompt,
        )
        self.add(model_name, "model_parameters", model_response.model_parameters)
        self.add(model_name, "llm_response", model_response.llm_response)
        self.add(model_name, "raw_response", model_response.raw_response)
        self.add(model_name, "tool_calls", getattr(model_response, 'tool_calls'))
        self.add(model_name, "raw_response_object", model_response.raw_response_object)
        self.add(model_name, "is_failure", model_response.is_failure)
        self.add(model_name, "is_empty", llm_response.strip() == "")
        self.add(model_name, "reasoning_response", model_response.reasoning_response)
        self.add(model_name, "reasoning_possible_to_extract", model_response.reasoning_possible_to_extract)
        self.add(model_name, "agent_messages", model_response.agent_messages)
        self.add(model_name, "reasoning_correct_format", model_response.reasoning_correct_format)
        self.add(model_name, "stop_reason", model_response.stop_reason)
        self.add(
            model_name,
            "relative_output_tokens",
            float(model_response.performance.relative_output_tokens)
            if model_response.performance
            else len(llm_response.split()),
        )
        if model_response.performance:
            input_tokens = model_response.performance.prompt_tokens
        elif isinstance(model_response.input_prompt, list):
            input_tokens = sum(len(text.split()) for text in self._iter_prompt_texts(model_response.input_prompt))
        else:
            input_tokens = len(model_response.input_prompt.split())
        self.add(model_name, "input_tokens", float(input_tokens))
        self.add(
            model_name,
            "output_tokens",
            float(model_response.performance.response_tokens)
            if model_response.performance
            else len(llm_response.split()),
        )
        self.add(
            model_name,
            "reasoning_tokens",
            model_response.performance.reasoning_tokens if model_response.performance else None,
        )
        self.add(model_name, "success", model_response.response_code == 200)
        self.add(
            model_name,
            "errors",
            model_response.error_tracker if model_response.error_tracker else model_response.response_code == 429,
        )
        self.add(
            model_name,
            "time_per_token",
            float(model_response.performance.time_per_token)
            if model_response.performance and model_response.performance.time_per_token
            else -1,
        )
        self.add(model_name, "response_code", model_response.response_code)

    def get_num_tokens(self, step) -> dict[str, float]:
        stats = self.statistics()
        if step == 0:
            for model_name, model_stats in stats.items():
                if all(k in model_stats for k in ("output_tokens", "relative_output_tokens", "reasoning_tokens")):
                    return {
                        f"{model_name}_average_output_tokens": model_stats["output_tokens"]["average"],
                        f"{model_name}_average_relative_output_tokens": model_stats["relative_output_tokens"]["average"],
                        f"{model_name}_average_reasoning_tokens": model_stats["reasoning_tokens"]["average"],
                    }
        else:
            step_str = str(step)
            for model_name, model_stats in stats.items():
                if step_str in model_name and all(k in model_stats for k in ("output_tokens", "relative_output_tokens", "reasoning_tokens")):
                    return {
                        f"{model_name}_average_output_tokens": model_stats["output_tokens"]["average"],
                        f"{model_name}_average_relative_output_tokens": model_stats["relative_output_tokens"]["average"],
                        f"{model_name}_average_reasoning_tokens": model_stats["reasoning_tokens"]["average"],
                    }
        return {}

    def _iter_prompt_texts(self, prompt: list[dict]):
        for message in prompt:
            match message:
                case {"content": str() as text}:
                    yield text
                case {"content": list() as content}:
                    for block in content:
                        match block:
                            case {"text": str() as text}:
                                yield text