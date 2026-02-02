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
        help="Path to agent YAML config file (default: config/agent.yaml)"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="config/env.yaml",
        help="Path to environment YAML config file (default: config/env.yaml)"
    )
    args = parser.parse_args()

    # Load agent config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    # Load environment config
    env_config_path = Path(args.env_config)
    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")

    env_config = load_config(env_config_path)

    # Extract config sections
    agent_cfg = config.get("agent", {})
    run_cfg = config.get("run", {})

    # Extract environment config sections
    game_cfg = env_config.get("game", {})
    env_cfg = env_config.get("env", {})
    info_requests_cfg = env_config.get("info_requests", {})

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

    # Create environment
    env = create_env(
        seed=game_cfg.get("seed"),
        quest_length=game_cfg.get("quest_length", 3),
        nb_rooms=game_cfg.get("nb_rooms", 3),
        nb_objects=game_cfg.get("nb_objects", 5),
        max_episode_steps=env_cfg.get("max_episode_steps", 100),
        intermediate_reward=env_cfg.get("intermediate_reward", True),
        info_requests=info_requests_cfg,
    )

    # Create runner
    runner = GameRunner(agent, env, verbose=not run_cfg.get("quiet", False))

    # Print setup info
    print(f"\n{'─'*60}")
    print(f"  TextWorld Agent")
    print(f"{'─'*60}")
    print(f"  Agent: {agent_type}")
    if agent_type in ("transformers", "openai"):
        model_name = agent_cfg.get("model", "default")
        print(f"  Model: {model_name}")
    print(f"  Objective mode: {agent_cfg.get('objective_mode', 'explicit')}")
    print(f"  Intermediate rewards: {env_cfg.get('intermediate_reward', True)}")
    print(f"  Episodes: {run_cfg.get('episodes', 1)}")
    print(f"{'─'*60}\n")

    # Run episodes
    episodes = run_cfg.get("episodes", 1)
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

    print(f"\n{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    print(f"  Episodes:      {len(results)}")
    print(f"  Average Score: {sum(scores)/len(scores):.2f}")
    print(f"  Average Steps: {sum(steps)/len(steps):.2f}")
    print(f"  Max Score:     {max(scores)}")
    print(f"{'─'*60}")

    env.close()


if __name__ == "__main__":
    main()
