from agent_arch.utils.util import (
    clean_id,
    get_registerable_tool_names_from_usecase,
    load_json_file,
)

TOOL_REGISTRY = {}


def register(name):
    """Register functions."""

    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func

    return decorator


def load_and_register_tools(usecase_config: dict):
    """Load tool configuration and register all custom tools dynamically"""

    filename = usecase_config["filename"]
    mock_data = load_json_file(f"configs/mocked_data/{filename}_mocked_tool_calls.json")

    all_tools = get_registerable_tool_names_from_usecase(usecase_config)

    for tool_name, tool_config in all_tools.items():

        def create_tool_func(name):
            def tool_func(*args, **kwargs):
                record_id = kwargs["internal_record_id"]
                return mock_data[clean_id(record_id)].get(name, "Not found")

            tool_func.__name__ = name

            docstring = tool_config.get("tool_description", f"Dynamic tool: {name}")

            if "input_parameters" in tool_config:
                docstring += "\n\nArgs:\n"
                for param_name, param_info in tool_config["input_parameters"].items():
                    docstring += f"    {param_name}: {param_info.get('description', 'No description')}\n"

            if "response_description" in tool_config:
                docstring += f"\nReturns:\n    {tool_config['response_description']}"

            tool_func.__doc__ = docstring
            return tool_func

        register(tool_name)(create_tool_func(tool_name))
