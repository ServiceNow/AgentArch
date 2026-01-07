# 🏗️ AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.10769-b31b1b.svg)](https://arxiv.org/abs/2509.10769)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

_A systematic evaluation framework for agentic AI systems across diverse architectural configurations and enterprise use cases._

</div>

---

## 🌟 Overview

AgentArch provides empirical insights into how different design dimensions interact within complex multi-agent systems. This benchmark evaluates **18 distinct agentic configurations** across state-of-the-art large language models, examining four critical system dimensions:

<table>
<tr>
<td>

### 🎯 **Orchestration Strategy**

Single-agent vs. multi-agent systems

</td>
<td>

### ⚙️ **Agent Implementation**

ReAct vs. function calling approaches

</td>
</tr>
<tr>
<td>

### 🧠 **Memory Architecture**

Complete vs. summarized memory management

</td>
<td>

### 🔧 **Thinking Tool Integration**

Mathematical reasoning and information synthesis tools

</td>
</tr>
</table>

---

## 🔍 Key Findings

> **TL;DR**: No one-size-fits-all solution exists for enterprise agentic systems

| Finding                           | Impact                                                                                                | 📊  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- | --- |
| **No Universal Architecture**     | Models demonstrate significant architectural preferences that vary by use case complexity             | 🎯  |
| **Performance Gaps**              | Even top models achieve only 35.3% success on complex enterprise tasks and 70.8% on simpler workflows | 📉  |
| **Multi-Agent ReAct Limitations** | Consistent underperformance across all models in multi-agent ReAct configurations                     | ⚠️  |
| **Reliability Challenges**        | Pass^K scores peak at only 6.34%, indicating fundamental gaps for production deployment               | 🚨  |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ServiceNow/AgentArch.git
cd AgentArch

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# 🔑 Replace placeholders with real API keys and endpoints
```

### Run Your First Evaluation

```python
python agent_arch/run.py \
  --mode single_agent \
  --usecase requesting_time_off \
  --model claude_sonnet_4 \
  --agent_type function_calling \
  --project test \
  --debug
```

### 🐳 Docker Evaluation

To run all evaluation configurations using Docker:

```bash
# Build the Docker image
docker build -f start.dockerfile -t agent-benchmark .

# Run the full benchmark suite
docker run --env-file .env \
  -e MODEL_TO_RUN=claude-sonnet-4 \
  -v $(pwd)/results:/app/results \
  agent-benchmark
```

**Environment Variables:**
- `MODEL_TO_RUN` - Required. The model to evaluate
- `PROJECT` - Project name for results (default: `default`)
- `BATCH_SIZE` - Batch size for evaluation (default: `70`)
- `DEBUG` - Enable debug mode (default: `false`)
- `K` - Pass@K trials count
- `SKIP_CONFIGS` - Comma-separated config numbers to skip

---

## 📁 Repository Structure

```
AgentArch/
├── 📁 agent_arch/                    # Main module
│   ├── 📁 configs/
│   │   ├── 🔧 mocked_data/
│   │   │   ├── customer_request_routing_mocked_tool_calls.json
│   │   │   └── requesting_time_off_mocked_tool_calls.json
│   │   ├── ⚙️ use_case_configs/
│   │   │   ├── customer_request_routing.yaml
│   │   │   └── requesting_time_off.yaml
│   │   └── 📜 prompts.yaml
│   ├── 📁 tools/
│   │   ├── base_agent_tools.json
│   │   ├── base_agent_tools.py
│   │   ├── thinking_tools.json
│   │   └── tool_registry.py
│   ├── 📁 utils/
│   │   ├── constants.py
│   │   ├── json_utils.py
│   │   ├── model.py
│   │   ├── model_response.py
│   │   ├── pass_k_metrics.py
│   │   ├── perf_stats.py
│   │   ├── run_context.py
│   │   └── util.py
│   ├── 🤖 agent.py
│   ├── 📊 metrics.py
│   └── ▶️ run.py                     # Main execution script
├── 📁 results/                        # Evaluation outputs
├── 🐳 start.dockerfile                # Docker image definition
├── 🐳 entrypoint.sh                   # Docker entrypoint script
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 requirements.txt
```

---

## 🏢 Enterprise Use Cases

<div align="center">

### 1. 📅 Requesting Time Off (TO) - Simple Workflow

</div>

| Aspect            | Details                                                          |
| ----------------- | ---------------------------------------------------------------- |
| **🎯 Complexity** | Basic multi-step reasoning with clear success criteria           |
| **🛠️ Tools**      | 8 custom enterprise tools                                        |
| **🤖 Agents**     | 3 specialized agents                                             |
| **💡 Challenges** | Date calculations, leave balance verification, policy compliance |

<div align="center">

### 2. 🎫 Customer Request Routing (CR) - Complex Workflow

</div>

| Aspect            | Details                                                         |
| ----------------- | --------------------------------------------------------------- |
| **🎯 Complexity** | Intelligent classification and escalation decisions             |
| **🛠️ Tools**      | 31 custom enterprise tools                                      |
| **🤖 Agents**     | 9 specialized agents                                            |
| **💡 Challenges** | Ambiguous request handling, context preservation, routing logic |

---

## 🤖 Evaluated Models

<div align="center">

| Provider      | Models                                 | Status |
| ------------- | -------------------------------------- | ------ |
| **OpenAI**    | GPT-4.1, GPT-4o, GPT-4.1-mini, o3-mini | ✅     |
| **Meta**      | LLaMA 3.3 70B                          | ✅     |
| **Anthropic** | Claude Sonnet 4                        | ✅     |

\*Framework includes support for evaluating any LiteLLM supported configuration

</div>

---

## 🏗️ Architectural Dimensions

### 🎭 Orchestration Strategies

#### 1. 🎪 Orchestrator-led, Isolated Agents

_Centralized task assignment with mediated communication_

<img width="400" alt="indirect" src="https://github.com/user-attachments/assets/64310502-3a7c-4bf6-b28c-d7330db36e70" />

#### 2. 🌐 Orchestrator-led, Open Network

_Initial task assignment with direct agent-to-agent communication_

<img width="400" alt="direct" src="https://github.com/user-attachments/assets/e1a2a174-4761-4925-b43f-b3a1de76a929" />

#### 3. 🤖 Single Agent

_Unified agent with access to all tools_

### 🎨 Agent Styles

<table>
<tr>
<td align="center">

### 📞 **Function Calling**

Direct tool selection using native model capabilities

</td>
<td align="center">

### 🧠 **ReAct**

Structured reasoning-action framework with explicit thought processes

</td>
</tr>
</table>

### 💾 Memory Management

<table>
<tr>
<td align="center">

### 📚 **Complete Memory**

Full visibility into all previous tool calls and responses

</td>
<td align="center">

### 📝 **Summarized Memory**

Condensed information sharing to manage context length

</td>
</tr>
</table>

### 🧮 Thinking Tools

<table>
<tr>
<td align="center">

### ➕ **Math Tool**

Structured mathematical reasoning and calculations

</td>
<td align="center">

### 🔍 **Synthesis Tool**

Information organization and analysis capabilities

</td>
</tr>
</table>

---

## 📊 Evaluation Metrics

### 🎯 Primary Metric: Acceptable Score

Success requires **simultaneous** achievement of:

- ✅ Correct tool selection
- ✅ Accurate tool arguments (100% accuracy required)
- ✅ Correct final decision

### 🔄 Reliability Metrics

- **Pass@1**: Success rate over k=8 trials
- **Pass^K**: Probability of all k trials succeeding

### 📈 Behavioral Metrics

- 🚫 Hallucination rates (non-existent tool/agent selection)
- 🔄 Tool repetition rates
- ❌ Missing required tools

---

## 💡 Key Recommendations

<div align="center">

### 👨‍💼 For Practitioners

</div>

| Recommendation                                 | Rationale                                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| ❌ **Avoid Multi-Agent ReAct**                 | Poor performance across all tested models                                                |
| ✅ **Use Multi-Agent for Final Decisions**     | Higher accuracy in decision-making despite tool selection challenges                     |
| 🎯 **Model-Specific Architectures**            | Test multiple configurations rather than assuming universal optima                       |
| 🧮 **Thinking Tools for Non-Reasoning Models** | Significant performance improvements on calculation-heavy tasks for non-reasoning models |

<div align="center">

### 🔬 For Researchers

</div>

| Focus Area                               | Insight                                                                             |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| 🔄 **Architecture-Use Case Interaction** | Models perform optimally under different architectures depending on task complexity |
| ⚖️ **Reliability vs Performance**        | Consider both accuracy and consistency for enterprise deployment                    |
| 💾 **Memory Management Impact**          | Minimal performance differences between complete and summarized memory              |

---

## 📚 Citation

```bibtex
@misc{bogavelli2025agentarchcomprehensivebenchmarkevaluate,
      title={AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise},
      author={Tara Bogavelli and Roshnee Sharma and Hari Subramani},
      year={2025},
      eprint={2509.10769},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.10769},
}
```

---

## 📄 License

AgentArch is licensed under the **Apache 2.0 License**.

## 📞 Contact

For questions or collaboration opportunities:

<div align="center">

[![Email](https://img.shields.io/badge/Email-tara.bogavelli%40servicenow.com-red?style=for-the-badge&logo=gmail)](mailto:tara.bogavelli@servicenow.com)

</div>

---

<div align="center">

**⭐ If this project helps your research, please consider giving it a star! ⭐**

</div>
