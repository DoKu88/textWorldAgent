# TextWorld AI Agent

A boilerplate for training AI agents on TextWorld environments using SFT, RL, and DPO.

## Setup

**Requirements:** Python 3.8 or higher (Python 3.9+ recommended)

```bash
pip install -r requirements.txt
```

## Usage

### Run with Random Agent
```bash
python main.py --agent random
```

### Run with LLM Agent
```bash
python main.py --agent llm
```

### Collect Training Data
```bash
python main.py --agent random --episodes 100 --collect-data --quiet
```

### Options
- `--agent`: Agent type (`random` or `llm`)
- `--model`: Model name for LLM agent (default: `google/flan-t5-small`)
- `--episodes`: Number of episodes to run
- `--max-steps`: Maximum steps per episode
- `--collect-data`: Save trajectories for training
- `--quiet`: Suppress verbose output

## Project Structure

```
├── main.py                 # Entry point
├── src/
│   ├── agent.py           # Agent implementations
│   ├── environment.py     # TextWorld env wrapper
│   ├── runner.py          # Game execution loop
│   └── data_collector.py  # Training data collection
├── games/                  # Generated game files
└── data/                   # Collected training data
```

## Training Data Formats (WIP)

- **SFT**: `data/sft_data.jsonl` - prompt/completion pairs
- **RL**: `data/rl_data.jsonl` - trajectories with rewards
- **DPO**: `data/dpo_data.jsonl` - chosen/rejected pairs
