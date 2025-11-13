from agent_arch.tools.tool_registry import register


@register("finish")
def finish(message: str, internal_record_id: str):
    """Finish mock tool call"""
    return {"message": message}


@register("synthesize_collected_information")
def synthesize_collected_information(
    synthesized_information: str, internal_record_id: str
):
    """Thinking tool for synthesizing collected information."""
    return synthesized_information


@register("math")
def math(input: str, internal_record_id: str):
    """Thinking tool for doing math."""
    return input
