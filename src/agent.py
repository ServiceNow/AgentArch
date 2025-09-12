from typing import Any, Dict, List, Union
import json
from tools.tool_registry import TOOL_REGISTRY, load_and_register_tools
from utils.util import load_prompt_messages, load_json_file, get_agents_as_tools, add_tool_call_requests_to_messages, parse_react_response, assign_tool_call_ids
from utils.constants import BUILT_IN_TOOLS, THINKING_TOOLS
from utils.run_context import RunContext
from utils.models import model_call
from utils.json_utils import extract_and_load_json
import dotenv
from tools import base_agent_tools
import copy
import math

AGENT_REGISTRY = {}
dotenv.load_dotenv()


def register(name):
    """Register functions."""

    def decorator(func):
        AGENT_REGISTRY[name] = func
        return func

    return decorator


class AgentRegistry:
    def __init__(self, config: dict):
        self.config = config
        self.agents = {}

    def register_agent(self, agent_name: str):
        """Register a single agent"""

        @register(agent_name)
        async def agent_function(internal_record_id: str, main_objective: str, mode: str, model_config: dict, thinking_tools_enabled: bool, agent_type: str, memory_type: str):
            agent = BaseAgent(
                agent_name=agent_name,
                usecase_config=self.config,
                mode=mode,
                model_config=model_config,
                thinking_tools_enabled=thinking_tools_enabled,
                agent_type=agent_type,
                memory_type=memory_type
            )
            return await agent.act(agent_objective=main_objective, internal_record_id=internal_record_id)

        self.agents[agent_name] = agent_function
        return agent_function

    def register_all_from_config_file(self):
        """Register all agents from a YAML config file"""

        agents_config = self.config["agents"]

        if not agents_config:
            raise ValueError(f"No 'agents' key found in config file")

        for agent_name in agents_config:
            self.register_agent(agent_name)


class BaseAgent:
    """
    Base class for all agents.
    """

    def __init__(
            self,
            agent_name: str,
            usecase_config: Dict[str, Union[Any, Dict[str, Any]]],
            model_config: dict,
            thinking_tools_enabled: bool,
            mode: str,
            agent_type: str,
            memory_type: str
    ):
        """
        Initialize the agent.

        Args:
            agent_name (str): Name of the agent.
            usecase_config (dict): Configuration for the usecase.
            model_config (dict): Configuration for the model.
            mode (str): Communication/orchestration mode for the agent.
            agent_type (str): Type of the agent either ReAct or Function calling
        """

        self.actions: List[str] = []
        self.agent_name = agent_name
        self.usecase_config = usecase_config
        self.agent_config = usecase_config["agents"].get(agent_name, {})
        self.instructions = self.set_agent_instructions()
        self.profile = self.agent_config.get("agent_profile", "")
        self.agent_type = agent_type
        self.mode = mode
        self.thinking_tools_enabled = thinking_tools_enabled
        self.tools = self.get_tools(mode, agent_name)
        self.model_config = model_config
        self.memory_type = memory_type
        # Register all tools for the use case
        load_and_register_tools(usecase_config)


    def set_agent_instructions(self):
        if self.agent_name != "single_agent":
            return self.agent_config["agent_instructions"]
        full_instructions = "You will be given high level instructions followed by step level instructions. Use the high level instructions to decide what steps to take (you must follow the order if steps are specified in order) and then to complete each high level step, follow the step level instructions. There may be no instructions in which case, use your best judgement on what to do next.\n\n"
        combined_instructions = "<begin_high_level_overall_instructions>\n\n" + self.usecase_config["use_case_instructions"] + "<end_high_level_overall_instructions>\n\n"+"Step level instructions:\n\n"
        full_instructions += combined_instructions
        for agent in self.usecase_config["agents"]:
            name = agent.rsplit("_", 1)[0]
            formatted_agent_instructions = f"<begin_{name}_instructions>\n" + self.usecase_config["agents"][agent]["agent_instructions"] + f"\n<end_{name}_instructions>\n\n"
            full_instructions += formatted_agent_instructions+ '\n'
        return full_instructions


    def get_tools(self, mode: str, agent_name) -> list[dict]:
        tools = []
        tool_schema = {
            "type": "function",
            "function":{
                "name": "",
                "description": "",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        }

        agents_to_process = (
            [agent_name] if agent_name != "single_agent" else list(self.usecase_config["agents"].keys())
        )

        for agent in agents_to_process:
            if self.usecase_config["agents"][agent].get("tools") is None:
                continue
            for tool in self.usecase_config["agents"][agent]["tools"]:
                current_tool = copy.deepcopy(tool_schema)
                current_tool["function"]["name"] = tool["tool_name"]
                current_tool["function"]["description"] = tool["tool_description"]

                tool_inputs = tool["input_parameters"]
                for input_name, input_schema in tool_inputs.items():
                    current_tool["function"]["parameters"]["properties"][input_name] = input_schema
                    if input_name not in current_tool["function"]["parameters"]["required"]:
                        current_tool["function"]["parameters"]["required"].append(input_name)

                tools.append(current_tool)

        base_tools = load_json_file("src/tools/base_agent_tools.json")
        tools.append(base_tools["finish"])
        if mode == "indirect":
            tools.append(base_tools["communicate_with_team"])
        elif mode == "direct":
            agents_and_proficiencies = []
            for agent in self.usecase_config["agents"]:
                if agent != agent_name:
                    agents_and_proficiencies.append({
                        "agent": agent,
                        "proficiency": self.usecase_config["agents"][agent]["agent_profile"]
                    })
            updated_description = base_tools["communicate_with_agent"].copy()["function"]["description"].replace("{{AGENT_PROFICIENCIES_PLACEHOLDER}}", json.dumps(agents_and_proficiencies))
            updated_communicate_with_agent =  base_tools["communicate_with_agent"].copy()
            updated_communicate_with_agent["function"]["description"] = updated_description
            tools.append(updated_communicate_with_agent)

        if self.thinking_tools_enabled:
            thinking_tools = load_json_file("src/tools/thinking_tools.json")
            for tool in thinking_tools:
                tools.append(thinking_tools[tool])

        return tools


    def load_react_prompt(self, agent_objective: str, tools: list[dict], internal_record_id: str):
        run_context = RunContext()
        messages = load_prompt_messages("agent_react",
        {
                    "agent_instructions": self.instructions,
                    "main_objective": agent_objective,
                    "memory": run_context.get_memory(internal_record_id),
                    "tools": json.dumps(self.get_react_formatted_tools(tools), indent=4)
                }
        )
        return messages

    def get_react_formatted_tools(self, tools: list):
        # might change formatting later
        return tools

    async def act(self, agent_objective: str, internal_record_id: str) -> Any:
        """
        Agent decides on an action to take.

        Args:
            task (str): The task to perform.

        Returns:
            Any: The action decided by the agent.
        """

        iterations = 0
        run_context = RunContext()

        messages = load_prompt_messages("agent", {
            "agent_instructions": self.instructions,
            "main_objective": agent_objective,
            "memory": run_context.get_memory(internal_record_id)
        })
        max_iterations = self.usecase_config["regular_max_iterations"] if self.agent_name != "single_agent" else self.usecase_config["single_agent_max_iterations"]

        while iterations <= max_iterations:
            model_name = self.model_config["name"]
            tools = self.tools
            model_response = await model_call(
                record_id=internal_record_id,
                model_config=self.model_config,
                model_name_with_call_id=f"{model_name}_{self.agent_name}_{iterations}",
                messages=messages if self.agent_type == "function_calling" else self.load_react_prompt(agent_objective, tools, internal_record_id),
                tools=tools if self.agent_type == "function_calling" else None,
                tool_choice="auto" if self.agent_type == "function_calling" else None
            )

            response_message = model_response.llm_response

            if self.agent_type == "function_calling":
                tool_calls = model_response.tool_calls
            else:
                tool_calls = parse_react_response(model_response.llm_response)

            if not tool_calls:
                run_context.add_to_memory(internal_record_id, {
                    "role": f"{self.agent_name}",
                    "content": response_message},
                    memory_type=self.memory_type
                )

                run_context.add_message_to_trace(
                    record_number=internal_record_id,
                    agent_name=self.agent_name,
                    content=response_message
                )
                break
            tool_calls = assign_tool_call_ids(tool_calls)

            # Add assistant response once
            # messages.append(model_response.raw_response_object.choices[0].message)
            if self.agent_type == "function_calling":
                messages = add_tool_call_requests_to_messages(self.model_config["model_family"], model_response, messages, tool_calls)

            tool_messages = []
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                args = json.loads(tool_args) if self.agent_type == "function_calling" else tool_args
                function_to_call = TOOL_REGISTRY.get(tool_name)


                run_context = RunContext()
                run_context.add_message_to_trace(
                    record_number=internal_record_id,
                    agent_name=self.agent_name,
                    content={
                        "tool_name": tool_name,
                        "tool_args": args,
                        "tool_call_id": tool_call["id"]
                    }
                )
                args.update({"internal_record_id": internal_record_id})
                if tool_name == "communicate_with_team" or tool_name == "communicate_with_agent":
                    run_context.add_to_memory(internal_record_id, {
                        "role": f"{self.agent_name}",
                        "content": {
                            "tool_name": tool_name,
                            "tool_args": args
                        }
                    },
                                              memory_type=self.memory_type)
                    if tool_name == "communicate_with_team":
                        try:
                            result = await self.communicate_with_team_flow(args["message"], internal_record_id)
                        except Exception as e:
                            result = f"Exception occurred: {e}"
                    elif tool_name == "communicate_with_agent":
                        try:
                            result = await self.communicate_with_agent_flow(args["agent"], args["message"], internal_record_id)
                        except Exception as e:
                            result = f"Exception occurred: {e}"
                elif function_to_call is None:
                    result = f"{tool_name} is not a valid tool"
                else:
                    try:
                        result = function_to_call(**args)
                    except Exception as e:
                        result = f"Exception occurred: {e}"


                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": json.dumps(result)
                })

                content_dict = {
                    "tool_name": tool_name,
                    "tool_args": args,
                    "tool_result": result,
                    "tool_call_id": tool_call["id"]
                }
                run_context.add_message_to_trace(
                    record_number=internal_record_id,
                    agent_name=self.agent_name,
                    content=content_dict
                )
                run_context.add_to_memory(
                    internal_record_id,
                    {
                        "role": f"{self.agent_name}",
                        "content": content_dict
                    },
                memory_type=self.memory_type
                )

                if tool_name == "finish" and isinstance(result, dict) and "message" in result:
                    # only exit if valid finish
                    return result.get("message", "")


            # Append all tool messages after the assistant message
            messages.extend(tool_messages)
            iterations += 1

        return "Agent loop exited after max iterations."

    async def communicate_with_team_flow(self, message: str, internal_record_id: str):
        """Communicate with team via orchestrator."""
        run_context = RunContext()
        agents = get_agents_as_tools(self.usecase_config)
        if self.agent_type == "function_calling":
            messages = load_prompt_messages("orchestrator_check_with_other_agents", {
                "agent_request": message
            })
        else:
            messages = load_prompt_messages("orchestrator_check_with_other_agents_REACT", {"request": message, "usecase_directions": self.usecase_config["use_case_instructions"], "memory": run_context.get_memory(record_number=internal_record_id), "agents":json.dumps(agents, indent=4)})

        agent_registry = AgentRegistry(self.usecase_config)
        agent_registry.register_all_from_config_file()
        model_response = await model_call(
            record_id=internal_record_id,
            model_config=self.model_config,
            model_name_with_call_id=f"{self.model_config['name']}_orchestrator_check_with_other_agents",
            messages=messages,
            tools=agents if self.agent_type == "function_calling" else None,
            tool_choice="auto" if self.agent_type == "function_calling" else None,
        )

        response_message = model_response.llm_response
        run_history = RunContext()
        if self.agent_type == "function_calling":
            agent_selections = model_response.tool_calls
        else:
            agent_selections = parse_react_response(response_message)
        if not agent_selections:
            run_history.add_message_to_trace(
                record_number=internal_record_id,
                agent_name="orchestrator",
                content=response_message,
            )
            return "Unfortunately no agents are able to help with this request."
        agent_selections = assign_tool_call_ids(agent_selections)
        response_to_agent = ""
        run_history.add_message_to_trace(
            record_number=internal_record_id,
            agent_name="orchestrator",
            content=[{"agent": agent["function"]["name"], "arguments":  json.loads(agent["function"]["arguments"]) if self.agent_type == "function_calling" else agent["function"]["arguments"]} for agent in agent_selections],
        )
        for agent in agent_selections:
            agent_name = agent["function"]["name"]
            agent_args = agent["function"]["arguments"]
            args = json.loads(agent_args) if self.agent_type == "function_calling" else agent_args
            agent_function = AGENT_REGISTRY.get(agent_name)
            if agent_function is None:
                continue
            args.update({"internal_record_id": internal_record_id, "model_config": self.model_config, "mode": "indirect", "thinking_tools_enabled": self.thinking_tools_enabled, "agent_type": self.agent_type, "memory_type": self.memory_type})
            try:
                result = await agent_function(**args)
            except Exception as e:
                print(f"exception occurred {str(e)}")
                result = f"Error calling agent {agent_name}: {e}"
            response_to_agent += str(result) + "\n"

        return response_to_agent

    async def communicate_with_agent_flow(self, requested_agent: str, message: str, internal_record_id: str):
        agent_name = requested_agent
        args = {
            "internal_record_id": internal_record_id,
            "main_objective": message,
            "mode": "direct",
            "thinking_tools_enabled": self.thinking_tools_enabled,
            "model_config": self.model_config,
            "agent_type": self.agent_type,
            "memory_type": self.memory_type
        }
        agent_function = AGENT_REGISTRY.get(agent_name)
        if agent_function is None:
            return "No registered function for agent: " + agent_name
        try:
            finish_message  = await agent_function(**args)
        except Exception as e:
            return f"Error calling agent {agent_name}: {e}"

        return finish_message
