# AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise

A systematic evaluation framework for agentic AI systems across diverse architectural configurations and enterprise use cases.

## Overview

AgentArch provides empirical insights into how different design dimensions interact within complex multi-agent systems. This benchmark evaluates 18 distinct agentic configurations across state-of-the-art large language models, examining four critical system dimensions:

- **Orchestration Strategy**: Single-agent vs. multi-agent systems
- **Agent Implementation**: ReAct vs. function calling approaches  
- **Memory Architecture**: Complete vs. summarized memory management
- **Thinking Tool Integration**: Mathematical reasoning and information synthesis tools

## Key Findings

- **No Universal Architecture**: Models demonstrate significant architectural preferences that vary by use case complexity
- **Performance Gaps**: Even top models achieve only 35.3% success on complex enterprise tasks and 70.8% on simpler workflows
- **Multi-Agent ReAct Limitations**: Consistent underperformance across all models in multi-agent ReAct configurations
- **Reliability Challenges**: Pass^K scores peak at only 6.34%, indicating fundamental gaps for production deployment

## Installation

```bash
git clone https://github.com/ServiceNow/AgentArch.git
cd AgentArch
pip install -r requirements.txt
cp .env.example .env
# replace placeholders with real api keys and endpoints
```

## Quick Start

```python
python -m src.run --mode single_agent --usecase requesting_time_off --model claude_sonnet_4 --agent_type function_calling --project test --debug
```


## Repository Structure

```
AgentArch/
├── configs/
│   ├── mocked_data/
│   │   ├── requesting_time_off_mocked_tool_calls.json
│   │   └── triage_cases_mocked_tool_calls.json
│   ├── use_case_configs/
│   │   ├── requesting_time_off.yaml
│   │   ├── triage_cases.yaml
│   │   └── prompts.yaml
├── results/ # sample results folder structure
│   └── test/ # project name
│       └── requesting_time_off/ # use case name
│           └── claude_sonnet_4/ # model name
│               └── single_agent/ # orchestration mode
│                   └── no_thinking/ # thinking 
│                       └── function_calling/ # function calling
│                           └── transparent/ # memory style
│                               └── 2025-09-12_13-02-08/ # timestamp
|                                   ├── metadata.json
|                                   ├── overall_scores.json
|                                   ├── perf_stats_overall.json
|                                   ├── record_level.csv
├── src/
│   ├── tools/            # Tool implementations
│   └── utils/
│       ├── agent.py      # Agent implementations
│       ├── metrics.py    # Evaluation metrics
│       └── run.py        # Main execution script
├── .env.example          # Environment template
├── .gitignore
└── requirements.txt
```

## Enterprise Use Cases

### 1. Requesting Time Off (TO) - Simple Workflow
- **Complexity**: Basic multi-step reasoning with clear success criteria
- **Tools**: 8 custom enterprise tools
- **Agents**: 3 specialized agents
- **Challenges**: Date calculations, leave balance verification, policy compliance

### 2. Customer Request Routing (CR) - Complex Workflow  
- **Complexity**: Intelligent classification and escalation decisions
- **Tools**: 31 custom enterprise tools
- **Agents**: 9 specialized agents
- **Challenges**: Ambiguous request handling, context preservation, routing logic

## Evaluated Models

- OpenAI GPT-4.1
- OpenAI GPT-4o  
- OpenAI GPT-4.1-mini
- OpenAI o3-mini
- LLaMA 3.3 70B
- Anthropic Claude Sonnet 4

## Architectural Dimensions

### Orchestration Strategies
1. **Orchestrator-led, Isolated Agents**: Centralized task assignment with mediated communication
   <img width="400" alt="indirect" src="https://github.com/user-attachments/assets/64310502-3a7c-4bf6-b28c-d7330db36e70" />
3. **Orchestrator-led, Open Network**: Initial task assignment with direct agent-to-agent communication
   <img width="400" alt="direct" src="https://github.com/user-attachments/assets/e1a2a174-4761-4925-b43f-b3a1de76a929" />
5. **Single Agent**: Unified agent with access to all tools


### Agent Styles
- **Function Calling**: Direct tool selection using native model capabilities
- **ReAct**: Structured reasoning-action framework with explicit thought processes

### Memory Management
- **Complete Memory**: Full visibility into all previous tool calls and responses
- **Summarized Memory**: Condensed information sharing to manage context length

### Thinking Tools
- **Math Tool**: Structured mathematical reasoning and calculations
- **Synthesis Tool**: Information organization and analysis capabilities

## Evaluation Metrics

### Primary Metric: Acceptable Score
Success requires simultaneous achievement of:
- Correct tool selection
- Accurate tool arguments (100% accuracy required)
- Correct final decision

### Reliability Metrics
- **Pass@1**: Success rate over k=8 trials
- **Pass^K**: Probability of all k trials succeeding

### Behavioral Metrics
- Hallucination rates (non-existent tool/agent selection)
- Tool repetition rates
- Missing required tools

## Key Recommendations

### For Practitioners
- **Avoid Multi-Agent ReAct**: Poor performance across all tested models
- **Use Multi-Agent for Final Decisions**: Higher accuracy in decision-making despite tool selection challenges
- **Model-Specific Architectures**: Test multiple configurations rather than assuming universal optima
- **Thinking Tools for Non-Reasoning Models**: Significant performance improvements on calculation-heavy tasks

### For Researchers
- **Architecture-Use Case Interaction**: Models perform optimally under different architectures depending on task complexity
- **Reliability vs Performance**: Consider both accuracy and consistency for enterprise deployment
- **Memory Management Impact**: Minimal performance differences between complete and summarized memory

## Citation


## License

## Contact

For questions or collaboration opportunities:
- Email: tara.bogavelli@servicenow.com
