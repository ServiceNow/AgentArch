from src.utils import constants
from src.utils.util import all_tool_configs, get_arg_eval_type
from src.utils import pass_k_metrics
import json
import re

def run_metrics(usecase_config: dict,record_history: list[dict], ground_truth: dict, mode: str, thinking_tools_enabled: bool) -> dict:
    """
    Computes scores for given record.
    :param usecase_config: Use case configuration
    :param record_history: Record trace
    :param ground_truth: Ground truth dictionary for the record
    :param mode: Communication/orchestrator mode
    :param thinking_tools_enabled: Flag to indicate if thinking tools are enabled
    :return: Score dictionary for given record.
    """

    agent_messages = condense_trace(record_history)

    actual_tools = []
    correct_tool_args_count = 0
    total_num_expected_args = 0
    number_of_thinking_tools_used = 0 if thinking_tools_enabled else "NA"
    number_of_custom_tools = 0
    final_message = ""
    actual_orchestrator_agent_selection = []
    exists_hallucination = False
    all_tools = all_tool_configs(usecase_config)

    tool_and_arg_tracker = []
    for idx, step_dict in enumerate(agent_messages):
        if step_dict["role"] == 'orchestrator':
            if isinstance(step_dict, dict) and "tool_args" in step_dict and isinstance(step_dict["tool_args"], list):
                for arg_dict in step_dict["tool_args"]:
                    if arg_dict["agent"] not in usecase_config["agents"]:
                        exists_hallucination = True
                    actual_orchestrator_agent_selection.append(arg_dict["agent"])
            if idx == len(agent_messages) - 1:
                final_message = step_dict["tool_args"] if isinstance(step_dict["tool_args"], str) else ""
        else:
            agent_name = step_dict["role"]
            if not "tool_name" in step_dict:
                continue
            tool_name = step_dict['tool_name']
            if tool_name not in all_tools and tool_name not in constants.THINKING_TOOLS and tool_name not in constants.BUILT_IN_TOOLS:
                exists_hallucination = True
            tool_and_arg_tracker.append({tool_name: step_dict["tool_args"]})
            if tool_name not in constants.BUILT_IN_TOOLS and tool_name not in constants.THINKING_TOOLS:
                actual_tools.append(tool_name)
            if tool_name in ground_truth["tool_args"]:
                for arg_name in ground_truth["tool_args"][tool_name]:
                    arg_val = step_dict["tool_args"].get(arg_name)
                    gt_arg_val = ground_truth["tool_args"][tool_name][arg_name]
                    if _match_ground_truth(arg_val, gt_arg_val) and correct_arg_type(arg_val, all_tools[tool_name]["input_parameters"][arg_name]["type"]):
                        correct_tool_args_count += 1
                    total_num_expected_args += 1

            if tool_name in constants.THINKING_TOOLS:
                number_of_thinking_tools_used += 1
            if tool_name not in constants.BUILT_IN_TOOLS and tool_name not in constants.THINKING_TOOLS:
                number_of_custom_tools += 1
            if tool_name == "finish" and mode == "single_agent":
                final_message = step_dict["tool_args"]["message"]

    correct_final_outcome = _get_correct_final_outcome(ground_truth, final_message)
    correct_tool_order, contains_extraneous_custom_tools, missing_expected_custom_tools = _tool_order_metrics(actual_tools, ground_truth)
    repeat_custom_tool_calls, repeat_thinking_tool_calls = _repeat_tool_metrics(tool_and_arg_tracker)
    correct_orchestrator_agent_selection =  _orchestrator_metrics(actual_orchestrator_agent_selection, ground_truth, mode)

    record_score_dict = {
        "actual_orchestrator_agent_selection": actual_orchestrator_agent_selection,
        "expected_orchestrator_agent_selection": ground_truth["orchestrator"]["agent_selections"],
        "missing_tools": [tool for tool in [t for gt in ground_truth["tool_order"] for t in gt] if tool not in actual_tools],
        "extraneous_tools": [tool for tool in actual_tools if tool not in [t for gt in ground_truth["tool_order"] for t in gt]],
        "actual_tools": actual_tools,
        "expected_tools": ground_truth["tool_order"],
        "final_message": final_message,
        "correct_final_outcome": correct_final_outcome,
        "correct_tool_order": correct_tool_order,
        "lenient_correct_tool_order": _check_lenient_acceptable_tool_order(correct_tool_order, actual_tools, missing_expected_custom_tools, usecase_config, ground_truth),
        "contains_extraneous_custom_tools": contains_extraneous_custom_tools,
        "missing_expected_custom_tools": missing_expected_custom_tools,
        "optimal_path": True if len(actual_tools) == ground_truth["min_custom_tool_number"] else False,
        "percent_of_correct_tool_args": correct_tool_args_count/total_num_expected_args if total_num_expected_args > 0 else 0, # for all tools that also appeared in ground truth, how many had correct args
        "correct_orchestrator_agent_selection": correct_orchestrator_agent_selection, # NA if single agent mode
        "repeat_custom_tool_calls": repeat_custom_tool_calls,
        "repeat_thinking_tool_calls": repeat_thinking_tool_calls,
        "number_of_custom_tools": len(actual_tools),
        "number_of_thinking_tools_used": number_of_thinking_tools_used,
        "total_number_of_agent_tool_calls": len(tool_and_arg_tracker),
        "exists_hallucination": exists_hallucination,
        "exists_tool_repetition": True if repeat_custom_tool_calls or repeat_thinking_tool_calls else False
    }
    print(json.dumps(record_score_dict, indent=4, ensure_ascii=False))
    return record_score_dict

def correct_arg_type(arg, arg_type):
    if arg_type == "bool":
        return isinstance(arg, bool)
    elif arg_type == "array":
        return isinstance(arg, list)
    elif arg_type == "object":
        return isinstance(arg, dict)
    elif arg_type == "int":
        return isinstance(arg, int)
    else:
        return isinstance(arg, str)

def _match_ground_truth(actual, gt) -> bool:
    if isinstance(actual, bool) or isinstance(gt, bool):
        return str(actual).lower() == str(gt).lower()
    if isinstance(actual, dict) and isinstance(gt, dict):
        return actual == gt
    if isinstance(actual, list) and isinstance(gt, list):
        # order doesn't matter
        norm_actual = {normalize_string(item) for item in actual}
        norm_gt = {normalize_string(item) for item in gt}
        return norm_actual == norm_gt
    return normalize_string(actual) == normalize_string(gt)

def _get_correct_final_outcome(ground_truth: dict, final_message: str) -> bool:
    if ground_truth["final_response"].lower() in final_message.lower():
        return True
    return False



def normalize_string(s):
    return re.sub(r'\s+', ' ', str(s).lower()).strip()

def _check_lenient_acceptable_tool_order(correct_tool_order, actual_tools, missing_expected_custom_tools, usecase_config, ground_truth):
    if correct_tool_order:
        return True
    if missing_expected_custom_tools:
        return False
    all_tools = all_tool_configs(usecase_config)
    all_tool_ground_truth_set = set(tool for tool_set in ground_truth["tool_order"] for tool in (tool_set if isinstance(tool_set, list) else [tool_set]))

    for tool in actual_tools:
        if tool not in all_tool_ground_truth_set and (tool in all_tools and all_tools[tool]["tool_type"] != "query"):
            return False
    return True


def _tool_order_metrics(actual_tools, ground_truth):
    """Gets tool order scores given actual tools and ground truth."""
    actual = actual_tools
    gts = ground_truth["tool_order"]
    correct_tool_order = True if any(actual == gt for gt in gts) else False
    gt_sets = [set(gt) for gt in gts]
    actual_set = set(actual)
    gt_union = set().union(*gt_sets)
    gt_intersection = set(gt_sets[0]).intersection(*gt_sets[1:]) if gt_sets else set()
    contains_extraneous_custom_tools = not actual_set.issubset(gt_union)
    missing_expected_custom_tools = not gt_intersection.issubset(actual_set)
    return correct_tool_order, contains_extraneous_custom_tools, missing_expected_custom_tools

def _repeat_tool_metrics(tool_and_arg_tracker):
    """Gets tool repeat metrics. Repeat is defined as tools with the same name and args as previous tool call."""
    prev = {}
    repeat_custom_tool_calls = 0
    repeat_thinking_tool_calls = 0
    for tool_call in tool_and_arg_tracker:
        tool_name = list(tool_call.keys())[0]
        if tool_call == prev and tool_name not in constants.THINKING_TOOLS:
            repeat_custom_tool_calls += 1
        elif tool_call == prev and tool_name in constants.THINKING_TOOLS:
            repeat_thinking_tool_calls += 1
        prev = tool_call
    return repeat_custom_tool_calls, repeat_thinking_tool_calls

def _orchestrator_metrics(actual_orchestrator_agent_selection, ground_truth, mode):
    """Gets orchestrator agent selection metrics."""
    if mode == "single_agent":
        return "NA"
    return True if any(actual_orchestrator_agent_selection == expected for expected in ground_truth["orchestrator"]["agent_selections"]) else False


def condense_trace(trace_data: list[dict]) -> list[dict]:
    """
    Condense a trace by combining tool calls with their results based on matching tool_call_id.
    Also includes orchestrator steps without tool_call_id.

    Args:
        trace_data: List of trace step dictionaries

    Returns:
        List of condensed dictionaries with agent, tool_name, tool_args, and tool_result (if available)
    """
    tool_calls = {}
    condensed_trace = []
    tool_call_indices = {}

    for step in trace_data:
        content = step.get('content', {})
        agent = step.get('agent')

        if agent == 'orchestrator':
            condensed_trace.append({
                'role': agent,
                'tool_name': None,
                'tool_args': content
            })
            continue

        # Handle non-dict content
        if not isinstance(content, dict):
            continue

        tool_call_id = content.get('tool_call_id')

        if tool_call_id:
            if tool_call_id not in tool_calls:
                # First occurrence - this is the tool call
                tool_call_entry = {
                    'role': agent,
                    'tool_name': content.get('tool_name'),
                    'tool_args': content.get('tool_args', {})
                }
                tool_calls[tool_call_id] = tool_call_entry
                condensed_trace.append(tool_call_entry)
                tool_call_indices[tool_call_id] = len(condensed_trace) - 1
            else:
                # Second occurrence - this has the result
                if 'tool_result' in content:
                    # Update the existing entry in the condensed trace
                    index = tool_call_indices[tool_call_id]
                    condensed_trace[index]['tool_result'] = content['tool_result']

    # Sort by step number to maintain order
    # condensed_trace.sort(key=lambda x: x['step'])

    return condensed_trace

def _get_overall_scores(results: list[dict], mode: str, thinking_tools_enabled: bool) -> dict:

    n = len(results)
    for r in results:
        if not isinstance(r, dict):
            print("here")
    scores = {
        "overall_acceptable_strict": sum(r["correct_final_outcome"] and not r["exists_hallucination"] and r["correct_tool_order"] and r["percent_of_correct_tool_args"] == 1.0 for r in results) / n,
        "overall_acceptable_lenient": sum(r["lenient_correct_tool_order"] and r["correct_final_outcome"] and r["percent_of_correct_tool_args"] == 1.0 for r in results) / n,
        "correct_tools_strict_rate": sum(r["correct_tool_order"] for r in results) / n,
        "correct_tools_lenient_rate": sum(r["lenient_correct_tool_order"] for r in results) / n,
        "correct_final_outcome_rate": sum(r["correct_final_outcome"] for r in results) / n,
        "extraneous_custom_tool_rate_↓": sum(r["contains_extraneous_custom_tools"] for r in results) / n,
        "missing_custom_tool_rate_↓": sum(r["missing_expected_custom_tools"] for r in results) / n,
        "hallucination_rate_↓":sum(r["exists_hallucination"] for r in results) / n,
        # "optimal_path_rate": sum(r["optimal_path"] for r in results) / n,
        "correct_tool_args_rate": sum(r["percent_of_correct_tool_args"] for r in results) / n,
        "tool_repetition_rate_↓": sum(r["exists_tool_repetition"] for r in results) / n,
        "avg_repeat_custom_tool_calls_↓": sum(r["repeat_custom_tool_calls"] for r in results) / n,
        "avg_total_number_of_agent_tool_calls_↓": sum(r["total_number_of_agent_tool_calls"] for r in results) / n,
    }
    if mode != "single_agent":
        scores["correct_orchestrator_rate"] = sum(r["correct_orchestrator_agent_selection"] and isinstance(r["correct_orchestrator_agent_selection"], bool) for r in results) / n
    if thinking_tools_enabled:
        scores.update({
            "avg_repeat_thinking_tool_calls_↓": sum(r["repeat_thinking_tool_calls"] for r in results) / n,
            "avg_number_of_thinking_tools_used_↓": sum(r["number_of_thinking_tools_used"] for r in results) / n,
        })
    return scores

def compute_overall_scores(results: list[dict], mode: str, thinking_tools_enabled: bool, pass_k: int) -> dict:
    """
    Computes overall scores based on record level results.

    :param results: List of record level results
    :param mode: Communication/orchestrator mode
    :param thinking_tools_enabled: Flag to indicate if thinking tools are enabled
    :param pass_k: Pass k value
    :return: Final scores dictionary
    """

    if not results:
        return {}

    if pass_k == 1:
        return _get_overall_scores(results, mode, thinking_tools_enabled)
    else:
        successful_list = [1 if r["correct_final_outcome"] and r["lenient_correct_tool_order"] and r["percent_of_correct_tool_args"] == 1.0 else 0 for r in results]
        scores = pass_k_metrics.PassAtKMetrics.calc_pass_k_metrics(successful_list, pass_k)
        scores.update(_get_overall_scores(results, mode, thinking_tools_enabled))
    return scores




