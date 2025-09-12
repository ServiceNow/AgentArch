import json
import yaml
from google.genai.types import Part
import os
import uuid
import csv
import ast
import pandas as pd
import shutil
import datetime
import re
from src.utils.perf_stats import PerfStats

from src.utils.json_utils import extract_and_load_json
ATTEMPT_PREFIX_PATTERN = re.compile(r'^(\d+_attempt_\d+)_')


def convert_message_history_to_tool_history(list_of_messages: list) -> list:
    message_history = []
    for message in list_of_messages:
        if isinstance(message, dict) and message.get("role", "") == "tool":
            message_history.append({
                "tool_name": message.get("name", ""),
                "content": message.get("content", "")
            })
    return message_history

def load_json_file(filepath: str):
    """Load data from a JSON file."""
    try:
        directory = os.getenv("DIRECTORY")
        filepath = os.path.join(directory, filepath)
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def load_yaml_file(filepath: str):
    """Load data from a YAML file."""
    try:
        directory = os.getenv("DIRECTORY")
        filepath = os.path.join(directory, filepath)
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

def load_prompt_messages(prompt_path: str, variables: dict) -> list:
    directory = os.getenv("DIRECTORY")
    path = os.path.join(directory, "configs/prompts.yaml")
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    keys = prompt_path.split(".")
    prompt_section = prompts
    for key in keys:
        prompt_section = prompt_section[key]

    messages = []
    for role, template in prompt_section.items():
        messages.append({
            "role": role,
            "content": template.format(**variables)
        })

    return messages

def get_agents_as_tools(usecase_config: dict) -> list:
    """Takes usecase config and returns all agents in the usecase as tools for orchestrator to select from."""
    agents = usecase_config["agents"]
    agents_as_tool_json = []
    for agent in agents:
        agent_as_tool = {
            "type": "function",
            "function": {
                "name": agent,
                "description": agents[agent]["agent_profile"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "main_objective": {
                            "type": "string",
                            "description": "The main objective or query for the leave approval agent to process."
                        }
                    },
                    "required": ["main_objective"],
                    "additionalProperties": False
                }
            }
        }
        agents_as_tool_json.append(agent_as_tool)
    return agents_as_tool_json

def clean_id(raw_id: str) -> str:
    return raw_id.split('_')[0]

def add_tool_call_requests_to_messages(model_family, model_response, messages, tool_call_requests):
    if "openai" in model_family or "qwen" in model_family:
        messages.append(model_response.raw_response_object.choices[0].message)
    elif "gemini" in model_family:
        function_call_parts = [
            Part.from_function_call(
                name=tool_call["function"]["name"],
                args=json.loads(tool_call["function"]["arguments"])
            )
            for tool_call in tool_call_requests
        ]

        model_tool_call_message = {
            "role": "model",
            "parts": function_call_parts
        }
        messages.append(model_tool_call_message)
    elif "anthropic" in model_family:
        tool_use_contents = [
            {
                "toolUse": {
                    "toolUseId": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"]),
                }
            }
            for tool_call in tool_call_requests
        ]

        messages.append({
            "role": "assistant",
            "content": tool_use_contents
        })
    return messages

def get_registerable_tool_names_from_usecase(usecase_config: dict) -> dict:
    all_tools = {}
    if 'agents' in usecase_config:
        for agent_name, agent_config in usecase_config['agents'].items():
            if 'tools' in agent_config:
                if not agent_config['tools']:
                    continue
                for tool in agent_config['tools']:
                    if tool.get("skip_automatic_tool_registration", False):
                        continue
                    tool_name = tool['tool_name']
                    all_tools[tool_name] = tool

    return all_tools

def all_tool_configs(usecase_config: dict) -> dict:
    all_tools = {}
    if 'agents' in usecase_config:
        for agent_name, agent_config in usecase_config['agents'].items():
            if 'tools' in agent_config:
                for tool in agent_config['tools']:
                    tool_name = tool['tool_name']
                    all_tools[tool_name] = tool

    return all_tools


def parse_react_response(response: str) -> list:
    if not is_valid_react_output(response):
        return []
    valid_react_output_json = extract_and_load_json(response)
    tools = []
    for tool_dict in valid_react_output_json:
        tool_name = tool_dict["action"]["name"]
        tool_args = tool_dict["action"]["arguments"]
        tool_dict = {
            "function": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        tools.append(tool_dict)

    return tools

def is_valid_react_output(response: str) -> bool:
    json_response = extract_and_load_json(response)
    if not json_response or not isinstance(json_response, list):
        return False
    for tool_dict in json_response:
        if not isinstance(tool_dict, dict):
            return False
        if list(tool_dict.keys()) != ["thought", "action", "observation"]:
            return False
        if not isinstance(tool_dict["thought"], str):
            return False
        if not isinstance(tool_dict["action"], dict):
            return False
        if "name" not in tool_dict["action"]:
            return False
        if not isinstance(tool_dict["action"]["name"], str):
            return False
        if "arguments" not in tool_dict["action"]:
            return False
        if not isinstance(tool_dict["action"]["arguments"], dict):
            return False

    return True

def assign_tool_call_ids(tool_calls: list[dict]) -> list[dict]:
    """Assign unique IDs to tool calls that don't have them."""
    for tool_call in tool_calls:
        if not tool_call.get("id") or tool_call.get("id", "") == "test":
            tool_call["id"] = f"call_{uuid.uuid4().hex[:12]}"
    return tool_calls

def get_arg_eval_type(tool_name,arg_name, agent_name, usecase_config) -> str:
    # tools = usecase_config["agents"][agent_name]["tools"]
    # for tool in tools:
    #     if tool["tool_name"] == tool_name:
    #         for arg in tool["input_parameters"]:
    #             if arg == arg_name:
    #                 return arg.get("eval_type", "")
    return ""

def remove_additional_properties(obj):
    if isinstance(obj, dict):
        # Remove the key if present
        obj.pop("additionalProperties", None)
        # Recurse into all nested values
        for value in obj.values():
            remove_additional_properties(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_additional_properties(item)

def get_records_to_rerun(results_dir: str, model_name: str) -> {}:
    """Load perf stats from an existing run."""
    csv_file = os.path.join(results_dir, f"perf_stats_overall_{model_name}_record_level.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"No perf stats file {csv_file} found in {results_dir}")
    needs_to_be_rerun = set()
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if _meets_criteria_for_rerun(row):
                cleaned_id = ATTEMPT_PREFIX_PATTERN.match(row["id"]).group(1) if "attempt" in row["id"] else row["id"].split("_")[0]
                needs_to_be_rerun.add(cleaned_id)
    return needs_to_be_rerun

def _meets_criteria_for_rerun(row: dict):
    if row["is_failure"] in ["True", "true", True, "TRUE"]:
        return True
    if "Exceeded maximum retry attempts" in str(row["raw_response"]):
        return True
    if str(row["llm_response"])=="" and str(row["raw_response"])=="" and str(row["raw_response_object"])=="":
        return True
    if "rate limit" in str(row["raw_response"]):
        return True
    return False



def load_existing_results(results_dir: str):
    """
    Load the trace data from an existing run.

    Args:
        results_dir: Directory containing the previous run results

    Returns:
        List of result dictionaries with traces
    """
    csv_file = os.path.join(results_dir, "record_level.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"No record_level.csv found in {results_dir}")

    results = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse the rows
            try:
                row['trace'] = ast.literal_eval(row['trace'])
                try:
                    row['final_memory'] = json.loads(row['final_memory'])
                except json.JSONDecodeError:
                    row['final_memory'] = ast.literal_eval(row['final_memory'])
            except (ValueError, TypeError, SyntaxError) as e:
                print(f"Error parsing row {row}: {e}")
            results.append(row)

    return results

def find_most_recent_matching_run(results_dir_pattern, current_timestamp):
    """
    Find the most recent run with the same configuration (excluding timestamp).

    Args:
        results_dir_pattern: Path pattern up to but not including timestamp
        current_timestamp: The current timestamp to exclude

    Returns:
        Path to the most recent matching directory, or None if not found
    """
    if not os.path.exists(results_dir_pattern):
        return None

    timestamp_dirs = []
    for item in os.listdir(results_dir_pattern):
        item_path = os.path.join(results_dir_pattern, item)
        if os.path.isdir(item_path) and item != current_timestamp:
            try:
                datetime.datetime.strptime(item, "%Y-%m-%d_%H-%M-%S")
                # Check if directory has all required files
                required_files = [
                    os.path.join(item_path, "record_level.csv")
                ]
                if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required_files):
                    timestamp_dirs.append(item)
            except ValueError:
                continue

    if not timestamp_dirs:
        return None

    # Sort by timestamp (most recent first)
    timestamp_dirs.sort(reverse=True)

    # Return the most recent one with actual results
    return os.path.join(results_dir_pattern, timestamp_dirs[0])

def copy_perf_stats_files(source_dir, dest_dir):
    """Copy all perf stats files from source to destination directory."""
    copied_files = []
    for file in os.listdir(source_dir):
        if file.startswith("perf_stats"):
            src = os.path.join(source_dir, file)
            dst = os.path.join(dest_dir, file)
            shutil.copy2(src, dst)
            copied_files.append(file)
    return copied_files

def collect_existing_perf_stats(source_dir, rerun_record_ids):
    """
    Collect existing perf stats data for records that were NOT rerun.
    Returns the data in the format expected by PerfStats.
    """
    existing_perf_data = {}

    # Look for CSV files in the source directory
    for file in os.listdir(source_dir):
        if file.startswith("perf_stats") and file.endswith("_record_level.csv"):
            src_file = os.path.join(source_dir, file)

            try:
                # Extract model name from filename (e.g., "perf_stats_overall_gpt-4o_record_level.csv")
                model_name = file.replace("perf_stats_", "").replace("_record_level.csv", "")
                if model_name.startswith("overall_"):
                    model_name = model_name.replace("overall_", "")

                # Load the CSV
                df = pd.read_csv(src_file)

                # Filter out records that were rerun
                def should_keep_record(record_id):
                    # Extract the base record ID to match against rerun_record_ids
                    record_id_str = str(record_id)
                    # Handle complex ID formats
                    parts = record_id_str.split('_')
                    if len(parts) >= 3 and parts[1] == 'attempt':
                        base_id = f"{parts[0]}_attempt_{parts[2]}"
                    else:
                        base_id = parts[0]
                    return base_id not in rerun_record_ids

                # Filter the dataframe
                if 'id' in df.columns:
                    filtered_df = df[df['id'].apply(should_keep_record)]
                else:
                    # Assume first column is the ID
                    filtered_df = df[df.iloc[:, 0].apply(should_keep_record)]

                # Convert back to the format that PerfStats expects
                existing_perf_data[model_name] = []
                for _, row in filtered_df.iterrows():
                    # Convert each row to a dict that matches what PerfStats.add() expects
                    row_dict = {}
                    for col in df.columns:
                        row_dict[col] = row[col]
                    existing_perf_data[model_name].append(row_dict)

                print(f"📋 Collected {len(filtered_df)} existing perf records for model {model_name}")

            except Exception as e:
                print(f"⚠️  Warning: Failed to collect existing perf stats from {file}: {e}")

    return existing_perf_data


def add_existing_perf_stats_to_current_session(existing_perf_data):
    """
    Add existing perf stats data to the current PerfStats singleton session.
    This merges the old data with the new data that was already added during run_in_batches.
    """
    perf_stats = PerfStats()

    for model_name, records in existing_perf_data.items():
        for record in records:
            # Add all metrics for this record as a single entry
            record_metrics = {}
            for metric, value in record.items():
                if pd.notna(value):  # Only add non-null values
                    record_metrics[metric] = value

            # Add this record as a single entry with all its metrics
            if record_metrics:
                perf_stats.stats[model_name].append(record_metrics)

def normalize_result_data_types(results: list[dict]) -> list[dict]:
    """
    Normalize data types in results to ensure consistency between
    newly run results and loaded results from previous runs.
    """
    # Define which fields should be boolean
    boolean_fields = [
        "correct_final_outcome",
        "exists_hallucination",
        "correct_tool_order",
        "lenient_correct_tool_order",
        "contains_extraneous_custom_tools",
        "missing_expected_custom_tools",
        "exists_tool_repetition",
        "correct_orchestrator_agent_selection"
    ]

    # Define which fields should be numeric
    numeric_fields = [
        "percent_of_correct_tool_args",
        "repeat_custom_tool_calls",
        "total_number_of_agent_tool_calls",
        "repeat_thinking_tool_calls",
        "number_of_thinking_tools_used"
    ]

    for result in results:
        # Convert boolean fields
        for field in boolean_fields:
            if field in result:
                value = result[field]
                if isinstance(value, str):
                    if value.lower() in ['true', 'TRUE', 'True', '1.0']:
                        result[field] = True
                    elif value.lower() in ['false', 'FALSE', 'False', '0.0']:
                        result[field] = False
                    else:
                        result[field] = False
                elif value is None:
                    result[field] = False

        # Convert numeric fields
        for field in numeric_fields:
            if field in result:
                value = result[field]
                if isinstance(value, str):
                    try:
                        result[field] = float(value)
                    except (ValueError, TypeError):
                        result[field] = 0.0
                elif value is None:
                    result[field] = 0.0

    return results

def _safe_load(object):
    if isinstance(object, (str, bytes, bytearray)):
        return json.loads(object)
    elif isinstance(object, dict):
        return object
    return {}
