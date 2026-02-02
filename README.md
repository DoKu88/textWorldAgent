# TextWorld AI Agent

A boilerplate for training AI agents on TextWorld environments using SFT, RL, and DPO.

## Setup

**Requirements:** Python 3.8 or higher (Python 3.9+ recommended)

```bash
pip install -r requirements.txt
```

## Usage

Run with default configuration:
```bash
python main.py
```

Use custom config files:
```bash
python main.py --config config/agent.yaml --env-config config/env.yaml
```

## Configuration

Configuration is managed through YAML files in the `config/` directory.

### Agent Configuration (`config/agent.yaml`)

```yaml
agent:
  type: openai                # Agent type: random, transformers, openai
  model: gpt-4o-mini          # Model name (for transformers/openai agents)
  history_length: 3           # Number of previous observations/actions to include
  objective_mode: abstract    # How much objective info: explicit, abstract, none

run:
  episodes: 10                # Number of episodes to run
  collect_data: false         # Collect trajectory data for training
  output: quiet               # quiet, normal, or verbose
```

### Environment Configuration (`config/env.yaml`)

```yaml
game:
  seed: null                  # Random seed for reproducibility (null = random)
  quest_length: 5             # Number of actions required to complete quest
  nb_rooms: 10                # Number of rooms in game world
  nb_objects: 7               # Number of objects in game world

env:
  max_episode_steps: 10       # Maximum steps before episode ends
  intermediate_reward: true   # Enable rewards for each sub-goal

info_requests:
  admissible_commands: true   # List of valid commands
  description: true           # Room descriptions
  inventory: true             # Player inventory
  max_score: true             # Maximum possible score
  won: true                   # Whether player has won
  lost: true                  # Whether player has lost
  objective: true             # Quest objective text
```

## Project Structure

```
├── main.py                 # Entry point
├── config/
│   ├── agent.yaml          # Agent and run configuration
│   └── env.yaml            # Environment and game configuration
├── src/
│   ├── agent.py            # Agent implementations
│   ├── environment.py      # TextWorld env wrapper
│   ├── runner.py           # Game execution loop
│   └── data_collector.py   # Training data collection
├── games/                  # Generated game files
└── data/                   # Collected training data
```

## Training Data Formats (WIP)

- **SFT**: `data/sft_data.jsonl` - prompt/completion pairs
- **RL**: `data/rl_data.jsonl` - trajectories with rewards
- **DPO**: `data/dpo_data.jsonl` - chosen/rejected pairs
