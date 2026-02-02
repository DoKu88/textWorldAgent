"""Main entry point for running TextWorld agent."""

import argparse
from pathlib import Path

import yaml

from src.agent import AgentFactory
from src.environment import create_env
from src.runner import GameRunner
from src.data_collector import TrajectoryCollector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run TextWorld AI Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config/agent.yaml",
        help="Path to YAML config file (default: config/agent.yaml)"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Extract config sections
    agent_cfg = config.get("agent", {})
    game_cfg = config.get("game", {})
    run_cfg = config.get("run", {})

    # Create agent using factory
    agent_type = agent_cfg.get("type", "random")
    agent_kwargs = {
        "history_length": agent_cfg.get("history_length", 0),
        "objective_mode": agent_cfg.get("objective_mode", "explicit"),
    }

    if agent_type == "transformers":
        agent_kwargs["model_name"] = agent_cfg.get("model", "google/flan-t5-small")
        agent_kwargs["verbose"] = not run_cfg.get("quiet", False)
    elif agent_type == "openai":
        agent_kwargs["model"] = agent_cfg.get("model", "gpt-4o-mini")
        agent_kwargs["verbose"] = not run_cfg.get("quiet", False)

    agent = AgentFactory.create(agent_type, **agent_kwargs)
    print(f"Using agent: {agent_type}")
    print(f"Objective mode: {agent_cfg.get('objective_mode', 'explicit')}")

    # Create environment
    print("Creating TextWorld environment...")
    env = create_env(
        seed=game_cfg.get("seed"),
        quest_length=game_cfg.get("quest_length", 3),
        nb_rooms=game_cfg.get("nb_rooms", 3),
        nb_objects=game_cfg.get("nb_objects", 5),
        max_episode_steps=run_cfg.get("max_steps", 50),
    )

    # Create runner
    runner = GameRunner(agent, env, verbose=not run_cfg.get("quiet", False))

    # Run episodes
    episodes = run_cfg.get("episodes", 1)
    print(f"\nRunning {episodes} episode(s)...\n")
    results = runner.run_episodes(episodes)

    # Collect data if requested
    if run_cfg.get("collect_data", False):
        collector = TrajectoryCollector()
        for score, steps, trajectory in results:
            collector.add_trajectory(trajectory)

        collector.save_sft_format()
        collector.save_rl_format()

    # Print summary
    scores = [r[0] for r in results]
    steps = [r[1] for r in results]

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Episodes: {len(results)}")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Average Steps: {sum(steps)/len(steps):.2f}")
    print(f"Max Score: {max(scores)}")

    env.close()


if __name__ == "__main__":
    main()
