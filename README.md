# Test-Time Search for Automated GFM Fine-Tuning


## Abstract

Graph Foundation Models (GFMs) have emerged as a powerful paradigm for learning transferable graph representations, yet adapting them to downstream tasks requires navigating an exponentially large decision space, traditionally demanding heavy expert effort. We propose **GFMTuner**, a framework that automates GFM fine-tuning by combining Large Language Model (LLM) agents with Monte Carlo Tree Search. GFMTuner accepts natural language task descriptions and generates effective fine-tuning strategies through test-time search. We introduce the **Graph-Instructed Actor**, which equips the LLM with graph analysis tools to ground action generation in structural insights, and **Gradient Consistency**, a self-supervised reward that measures gradient alignment across perturbed executions for efficient strategy evaluation. Experiments across diverse graph domains demonstrate that GFMTuner matches or exceeds human expert designs while reducing effort from weeks to a single natural language query.

---

## Quick Start

### 1. Environment Setup

```bash

# cd GFMTuner

# Use the supplementary code provided
cd GFMTuner

# Create conda environment
conda create -n GFMTuner python=3.10
conda activate GFMTuner

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys Or Deploy Locally

Create a `.env` file in the root directory:

```bash
# For OpenAI API
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_api_url  # Optional

# For Google Gemini (alternative)
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Generate Sample Tasks

```bash
python script/generate_GFMTuner_tasks.py
```

This creates `data/GFMTuner/tasks.pkl` with a sample graph learning task.

### 4. Run GFMTuner

```bash
conda run -n GFMTuner python -m gfmtuner.runner.mcts_runner config/GFMTuner_example.yaml
```

Results will be saved to `results/GFMTuner/demo/`.
