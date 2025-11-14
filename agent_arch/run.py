import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.getenv("DIRECTORY"):
    os.environ["DIRECTORY"] = BASE_DIR
os.environ["VERTEX_APPLICATION_CREDENTIALS"] = os.path.join(
    os.environ["DIRECTORY"], "vertex_credentials.json"
)


from agent_arch.agent import AGENT_REGISTRY, AgentRegistry, BaseAgent
from agent_arch.metrics import compute_overall_scores, run_metrics
from agent_arch.utils.model import model_call, get_model_info
from agent_arch.utils.perf_stats import PerfStats
from agent_arch.utils.run_context import RunContext
from agent_arch.utils.util import (
    add_existing_perf_stats_to_current_session,
    assign_tool_call_ids,
    collect_existing_perf_stats,
    copy_perf_stats_files,
    find_most_recent_matching_run,
    get_agents_as_tools,
    get_records_to_rerun,
    load_existing_results,
    load_prompt_messages,
    load_yaml_file,
    normalize_result_data_types,
    parse_react_response,
)

run_context = RunContext()


async def process_record(
    record,
    usecase_config: dict,
    model_config: dict,
    mode: str,
    thinking_tools_enabled: bool,
    agent_type: str,
    memory_type: str,
):
    """Runs the flow for a single record."""

    record_id = record["id"]
    usecase_directions = usecase_config["use_case_instructions"]
    run_context.add_to_memory(
        record_number=record_id,
        content={"role": "user", "content": record["user_utterance"]},
        memory_type=memory_type,
    )

    user_utterance = record["user_utterance"]

    messages = []
    if mode != "single_agent":
        messages = load_prompt_messages(
            "orchestrator_initial",
            {
                "user_utterance": user_utterance,
                "usecase_directions": usecase_directions,
                "memory": run_context.get_memory(record_number=record_id),
            },
        )

    max_iterations = usecase_config["orchestrator_max_iterations"]
    iterations = 0
    continue_loop = True

    while iterations < max_iterations and continue_loop:
        if "single_agent" in mode:
            passed_mode = "_".join(mode.split("_")[2:]) if mode.count("_") >= 2 else ""
            single_agent = BaseAgent(
                agent_name="single_agent",
                usecase_config=usecase_config,
                mode=passed_mode,
                model_config=model_config,
                thinking_tools_enabled=thinking_tools_enabled,
                agent_type=agent_type,
                memory_type=memory_type,
            )
            finish_message = await single_agent.act(
                agent_objective=user_utterance, internal_record_id=record_id
            )
            continue_loop = False

        else:
            agents = get_agents_as_tools(usecase_config)
            agent_registry = AgentRegistry(usecase_config)
            agent_registry.register_all_from_config_file()
            model_response = await model_call(
                record_id=record_id,
                model_config=model_config,
                model_name_with_call_id=f"{model_config['name']}_orchestrator_{iterations}",
                messages=messages
                if agent_type == "function_calling"
                else load_prompt_messages(
                    "orchestrator_react",
                    {
                        "request": user_utterance,
                        "usecase_directions": usecase_directions,
                        "memory": run_context.get_memory(record_number=record_id),
                        "agents": json.dumps(
                            get_agents_as_tools(usecase_config), indent=4
                        ),
                    },
                ),
                tools=agents if agent_type == "function_calling" else None,
                tool_choice="auto" if agent_type == "function_calling" else None,
            )

            response_message = model_response.llm_response
            if agent_type == "function_calling":
                agent_selections = model_response.tool_calls
            else:
                agent_selections = parse_react_response(response_message)

            if not agent_selections:
                run_context.add_to_memory(
                    record_number=record_id,
                    content={"role": "orchestrator", "content": response_message},
                    memory_type=memory_type,
                )
                run_context.add_message_to_trace(
                    record_number=record_id,
                    agent_name="orchestrator",
                    content=response_message,
                )

                messages.append(response_message)
                continue_loop = False
            else:
                agent_selections = assign_tool_call_ids(agent_selections)
                orchestrator_content = [
                    {
                        "agent": agent["function"]["name"],
                        "arguments": json.loads(agent["function"]["arguments"])
                        if agent_type == "function_calling"
                        else agent["function"]["arguments"],
                    }
                    for agent in agent_selections
                ]
                run_context.add_message_to_trace(
                    record_number=record_id,
                    agent_name="orchestrator",
                    content=orchestrator_content,
                )
                run_context.add_to_memory(
                    record_number=record_id,
                    content={"role": "orchestrator", "content": orchestrator_content},
                    memory_type=memory_type,
                )

                tool_result_messages = []
                for agent in agent_selections:
                    agent_name = agent["function"]["name"]
                    agent_args = (
                        json.loads(agent["function"]["arguments"])
                        if agent_type == "function_calling"
                        else agent["function"]["arguments"]
                    )

                    agent_function = AGENT_REGISTRY.get(agent_name)
                    if agent_function is None:
                        continue

                    agent_args.update(
                        {
                            "internal_record_id": record_id,
                            "mode": mode,
                            "model_config": model_config,
                            "thinking_tools_enabled": thinking_tools_enabled,
                            "agent_type": agent_type,
                            "memory_type": memory_type,
                        }
                    )
                    try:
                        finish_message = await agent_function(**agent_args)
                    except Exception as e:
                        finish_message = f"Exception occurred: {e}"

                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": agent["id"],
                            "name": agent_name,
                            "content": finish_message,
                        }
                    )
                messages.append(model_response.raw_response_object.choices[0].message)
                messages.extend(tool_result_messages)
        iterations += 1

    trace = run_context.get_record_trace(record_id)
    scores = run_metrics(
        usecase_config, trace, record["ground_truth"], mode, thinking_tools_enabled
    )

    return {
        "id": record_id,
        "user_utterance": user_utterance,
        "trace": trace,
        "final_memory": json.dumps(
            run_context.get_memory(record_id), ensure_ascii=False
        ),
        **scores,
    }


def get_records(records, k):
    records = records[:2]
    if k == 1:
        return records
    augmented_records = []
    for record in records:
        for i in range(k):
            new_id = record["id"] + f"_attempt_{i}"
            added_record = record.copy()
            added_record["id"] = new_id
            augmented_records.append(added_record)
    return augmented_records


async def run_in_batches(
    records,
    usecase_config,
    model_config,
    mode,
    thinking_tools_enabled,
    agent_type,
    memory_type,
):
    batch_size = int(os.getenv("BATCH_SIZE", 70))
    print(f"Running in batches of {batch_size}")
    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        tasks = [
            process_record(
                record,
                usecase_config,
                model_config,
                mode,
                thinking_tools_enabled,
                agent_type,
                memory_type,
            )
            for record in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results


async def main(
    usecase: str,
    model: str,
    mode: str,
    debug: bool,
    thinking_tools_enabled: bool,
    pass_k: int,
    project: str,
    agent_type: str,
    memory_type: str,
    directory: str,
):
    if not directory:
        directory = os.getenv("DIRECTORY")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(
        directory,
        "results",
        project,
        usecase,
        model,
        mode,
        "thinking_enabled" if thinking_tools_enabled else "no_thinking",
        agent_type,
        memory_type,
        timestamp,
    )
    os.makedirs(results_dir, exist_ok=True)

    model_config = get_model_info(model)
    usecase_config = load_yaml_file(
        f"{directory}/configs/use_case_configs/{usecase}.yaml"
    )

    RERUN_METRICS = os.getenv("RERUN_METRICS", False)
    RERUN_ERROR_RECORDS = os.getenv("RERUN_ERROR_RECORDS", False)
    results = []
    if RERUN_METRICS or RERUN_ERROR_RECORDS:
        print(f"🔄 RERUN enabled - looking for existing results to re-run metrics on")
        results_dir_pattern = os.path.join(
            directory,
            "results",
            project,
            usecase,
            model,
            mode,
            "thinking_enabled" if thinking_tools_enabled else "no_thinking",
            agent_type,
            memory_type,
        )
        previous_run_dir = find_most_recent_matching_run(results_dir_pattern, timestamp)

        if not previous_run_dir:
            print(
                "❌ No previous run found with matching configuration! Try running regularly"
            )
            print(f"   Looked in: {results_dir_pattern}")
            return
        print(f"✅ Found previous run: {previous_run_dir}")
        if RERUN_ERROR_RECORDS:
            try:
                # ids_of_records_to_rerun is set of ids that need to be rerun
                model_config = json.loads(os.getenv(f"{model.upper()}_CONFIG"))
                ids_of_records_to_rerun = get_records_to_rerun(
                    previous_run_dir, model_config["name"]
                )
                print(
                    f"📊 {len(ids_of_records_to_rerun)} records from previous run need to be rerun"
                )

                existing_results = load_existing_results(previous_run_dir)
                print(
                    f"📋 Loaded {len(existing_results)} existing records from previous run"
                )

                existing_results_by_id = {
                    result["id"]: result for result in existing_results
                }
                records_to_rerun = []
                for record in usecase_config["utterances_and_ground_truths"]:
                    # Handle both pass_k format (id_attempt_X) and regular format
                    if pass_k > 1:
                        # For pass_k > 1, check if any attempt of this record needs rerunning
                        for attempt in range(pass_k):
                            record_id_with_attempt = f"{record['id']}_attempt_{attempt}"
                            if record_id_with_attempt in ids_of_records_to_rerun:
                                # Create a copy of the record with the attempt-specific ID
                                record_copy = record.copy()
                                record_copy[
                                    "id"
                                ] = record_id_with_attempt  # dont think this is needed, check
                                records_to_rerun.append(record_copy)
                    else:
                        # For pass_k = 1, direct ID matching
                        if record["id"] in ids_of_records_to_rerun:
                            records_to_rerun.append(record)

                print(f"🔄 Re-running {len(records_to_rerun)} specific records...")
                new_results = await run_in_batches(
                    records_to_rerun,
                    usecase_config,
                    model_config,
                    mode,
                    thinking_tools_enabled,
                    agent_type,
                    memory_type,
                )

                final_results = []
                new_results_by_id = {result["id"]: result for result in new_results}
                for existing_result in existing_results:
                    record_id = existing_result["id"]
                    if record_id in new_results_by_id:
                        # Use the new result
                        final_results.append(new_results_by_id[record_id])
                        print(f"✅ Replaced result for record: {record_id}")
                    else:
                        # Keep the existing result unchanged
                        final_results.append(existing_result)
                final_results = normalize_result_data_types(final_results)
                results = final_results
                print(f"✅ Final merged results contain {len(results)} records")
                existing_perf_data = collect_existing_perf_stats(
                    previous_run_dir, ids_of_records_to_rerun
                )
                add_existing_perf_stats_to_current_session(existing_perf_data)
                print(
                    f"📊 Merged perf stats: kept existing data for non-rerun records, new data for rerun records"
                )

            except Exception as e:
                print(f"❌ Error loading records to rerun: {e}")
                return

        elif RERUN_METRICS:
            try:
                results = load_existing_results(previous_run_dir)
                print(f"📊 Loaded {len(results)} records from previous run")

                # Re-run metrics on each result
                for i, result in enumerate(results):
                    trace = result["trace"]
                    record_id = result["id"]
                    user_utterance = result["user_utterance"]

                    # Find the corresponding ground truth
                    ground_truth = None
                    for record in usecase_config["utterances_and_ground_truths"]:
                        # Handle both original IDs and pass_k augmented IDs
                        base_id = (
                            record_id.split("_attempt_")[0]
                            if "_attempt_" in record_id
                            else record_id
                        )
                        if record["id"] == base_id or record["id"] == record_id:
                            ground_truth = record["ground_truth"]
                            break

                    if not ground_truth:
                        print(
                            f"⚠️  Warning: No ground truth found for record {record_id}"
                        )
                        continue

                    scores = run_metrics(
                        usecase_config,
                        trace,
                        ground_truth,
                        mode,
                        thinking_tools_enabled,
                    )
                    for key, value in scores.items():
                        result[key] = value

                    print(f"✅ Re-scored record {i+1}/{len(results)}: {record_id}")

                # copy perf stats over
                os.makedirs(results_dir, exist_ok=True)
                copied_files = copy_perf_stats_files(previous_run_dir, results_dir)
                if copied_files:
                    print(
                        f"📋 Copied {len(copied_files)} perf stats file(s) from previous run"
                    )

            except Exception as e:
                print(f"❌ Error loading previous results: {e}")
                return
    else:
        if debug:
            results = [
                await process_record(
                    usecase_config["utterances_and_ground_truths"][49],
                    usecase_config,
                    model_config,
                    mode,
                    thinking_tools_enabled,
                    agent_type,
                    memory_type,
                )
            ]
        else:
            records = get_records(
                usecase_config["utterances_and_ground_truths"], pass_k
            )
            results = await run_in_batches(
                records,
                usecase_config,
                model_config,
                mode,
                thinking_tools_enabled,
                agent_type,
                memory_type,
            )

    overall_scores = compute_overall_scores(
        results, mode, thinking_tools_enabled, pass_k
    )
    perf_stats = PerfStats()
    perf_file = os.path.join(results_dir, f"perf_stats_overall")
    perf_stats.generate_summary(perf_file)
    # load failure rates from perf stats file and add to overall scores
    json_path = f"{perf_file}.json"
    overall_perf_stats_dict = json.load(open(json_path, "r"))
    failure_rates = []
    for model_key in overall_perf_stats_dict:
        avg_failure_rate = (
            overall_perf_stats_dict[model_key].get("is_failure", {}).get("average", 0)
        )
        if avg_failure_rate:
            failure_rates.append(avg_failure_rate)
    if failure_rates:
        overall_scores.update(
            {
                "failure_rate": sum(failure_rates) / len(failure_rates)
                if len(failure_rates) > 0
                else 0
            }
        )
    csv_file = os.path.join(results_dir, "record_level.csv")
    with open(csv_file, "w", newline="") as file:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_file = os.path.join(results_dir, "overall_scores.json")
    with open(json_file, "w") as f:
        json.dump(overall_scores, f, indent=4)

    duration = time.time() - time.mktime(
        datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S").timetuple()
    )
    print(f"\n✅ Finished processing {len(results)} records in {duration:.2f} seconds.")
    print("\n📈 Overall Scores...")
    print(json.dumps(overall_scores, indent=4, ensure_ascii=False))
    metadata_file = os.path.join(results_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        metadata = {
            "usecase": usecase,
            "model": model,
            "mode": mode,
            "thinking_tools_enabled": thinking_tools_enabled,
            "pass_k": pass_k,
            "time_taken": f"{duration:.2f}",
        }
        if RERUN_METRICS:
            metadata["rerun_metrics"] = True
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Choose model to run")
    parser.add_argument("--usecase", required=True, help="Which usecase to run")
    parser.add_argument(
        "--mode",
        choices=[
            "direct",
            "indirect",
            "single_agent",
            "single_agent_RAG_tool_curation",
            "single_agent_LLM_tool_curation",
        ],
        required=True,
        default="direct",
        help="Choose mode: direct communication, indirect communication or single agent mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--thinking-tools-enabled", action="store_true", help="Enable thinking tools"
    )
    parser.add_argument(
        "--pass_k",
        default=1,
        type=int,
        help="Choose pass k value to compute pass@k and pass^k",
    )
    parser.add_argument(
        "--project", required=False, default="output", help="Project name"
    )
    parser.add_argument(
        "--agent_type",
        choices=["function_calling", "ReAct"],
        required=False,
        default="function_calling",
        help="Either function_calling or ReAct",
    )
    parser.add_argument(
        "--memory_management",
        choices=["transparent", "compact"],
        required=False,
        default="transparent",
        help="Either transparent or compact",
    )
    parser.add_argument(
        "--directory",
        required=False,
        default="",
        help="Parent directory where results directory will be created",
    )
    args = parser.parse_args()
    if args.memory_management == "compact" and args.agent_type == "ReAct":
        raise ValueError("ReAct only supports transparent memory management")
    asyncio.run(
        main(
            args.usecase,
            args.model,
            args.mode,
            args.debug,
            args.thinking_tools_enabled,
            args.pass_k,
            args.project,
            args.agent_type,
            args.memory_management,
            args.directory,
        )
    )
