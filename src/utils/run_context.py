from typing import Any, Optional
import datetime

class RunContext:
    """
    Singleton class to store run contexts for each record.
    Each record is stored by its record_number, with a list of agent outputs.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RunContext, cls).__new__(cls)
            cls._instance.traces: dict[str, list[dict[str, Any]]] = {}
            cls._instance.memories:  dict[str, list[dict[str, Any]]] = {}
            cls._instance.step_counter: dict[str, int] = {}
        return cls._instance

    
    def add_message_to_trace(self, record_number: str, agent_name: str, content: Any) -> None:
        """
        Add a new message to the conversation history for a specific record.
        
        Args:
            record_number: The unique identifier for the record
            agent_name: Name of the agent adding the message
            content: The content of the message
            metadata: Optional additional data to store with the message
        """
        if record_number not in self.traces:
            self.traces[record_number] = []


        step = self.step_counter.get(record_number, 0)
        self.step_counter[record_number] = step + 1
            
        entry = {
            "step": step,
            'agent': agent_name,
            'content': content,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        self.traces[record_number].append(entry)
    
    def get_record_trace(self, record_number: str) -> list[dict[str, Any]]:
        """
        Get the conversation history for a specific record.
        
        Args:
            record_number: The record number to get history for
            
        Returns:
            List of message dictionaries for the specified record, or empty list if no history exists
        """
        trace = self.traces.get(record_number, [])
        return sorted(trace, key=lambda x: x["step"])


    def add_to_memory(self, record_number: str, content: dict, memory_type: str) -> None:
        """
        Add a new step to the memory for a specific record.

        Args:
        :param record_number: Record number
=       :param content: Content dictionary to add to memory.
        :return:
        """
        if record_number not in self.memories:
            self.memories[record_number] = []

        role = content.get("role", "")
        content_data = content.get("content", {})

        if memory_type == "compact":
            # only append if finish
            if role != "orchestrator" and role != "user":
                if isinstance(content_data, dict):
                    tool_name = content_data.get("tool_name", "")
                    tool_result = content_data.get("tool_result", {})
                    message = tool_result.get("message")

                    if tool_name == "finish" and isinstance(tool_result, dict) and message is not None:
                        self.memories[record_number].append({
                            "role": role,
                            "content": message,
                        })
            elif role != "orchestrator":
                self.memories[record_number].append(content)
        else:
            self.memories[record_number].append(content)

    def get_memory(self, record_number: str) -> list:
        """
        Get memory for specific record.
        :param record_number:
        :return: Memory
        """
        return self.memories.get(record_number, [])

