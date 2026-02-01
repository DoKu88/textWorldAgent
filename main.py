"""Main entry point for running TextWorld agent."""

import argparse

from src.agent import RandomAgent, LLMAgent
from src.environment import create_env
from src.runner import GameRunner
from src.data_collector import TrajectoryCollector


def main():
    parser = argparse.ArgumentParser(description="Run TextWorld AI Agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "llm"],
        help="Agent type to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="Model name for LLM agent (default: google/flan-t5-small)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect trajectory data for training"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Create agent
    if args.agent == "random":
        agent = RandomAgent()
        print("Using Random Agent")
    else:
        agent = LLMAgent(model_name=args.model)
        print(f"Using LLM Agent with model: {args.model}")

    # Create environment
    print("Creating TextWorld environment...")
    env = create_env(max_episode_steps=args.max_steps)

    # Create runner
    runner = GameRunner(agent, env, verbose=not args.quiet)

    # Run episodes
    print(f"\nRunning {args.episodes} episode(s)...\n")
    results = runner.run_episodes(args.episodes)

    # Collect data if requested
    if args.collect_data:
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
