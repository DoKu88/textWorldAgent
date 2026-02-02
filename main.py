"""Main entry point for running TextWorld agent."""

import argparse

from src.agent import AgentFactory
from src.environment import create_env
from src.runner import GameRunner
from src.data_collector import TrajectoryCollector


def main():
    parser = argparse.ArgumentParser(description="Run TextWorld AI Agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=AgentFactory.list_agents(),
        help=f"Agent type to use. Available: {', '.join(AgentFactory.list_agents())}"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: google/flan-t5-small for llm, gpt-4o-mini for openai)"
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
    parser.add_argument(
        "--history-length",
        type=int,
        default=0,
        help="Number of previous observations/actions to include in prompt (default: 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for game generation (for reproducibility)"
    )
    parser.add_argument(
        "--quest-length",
        type=int,
        default=3,
        help="Number of actions required to complete the quest (default: 3)"
    )
    parser.add_argument(
        "--nb-rooms",
        type=int,
        default=3,
        help="Number of rooms in the game world (default: 3)"
    )
    parser.add_argument(
        "--nb-objects",
        type=int,
        default=5,
        help="Number of objects in the game world (default: 5)"
    )

    args = parser.parse_args()

    # Create agent using factory with agent-specific defaults
    agent_kwargs = {"history_length": args.history_length}
    if args.agent in ["llm", "llm-transformers"]:
        agent_kwargs["model_name"] = args.model or "google/flan-t5-small"
        agent_kwargs["verbose"] = not args.quiet
    elif args.agent == "openai":
        agent_kwargs["model"] = args.model or "gpt-4o-mini"
        agent_kwargs["verbose"] = not args.quiet

    agent = AgentFactory.create(args.agent, **agent_kwargs)
    print(f"Using agent: {args.agent}")

    # Create environment
    print("Creating TextWorld environment...")
    env = create_env(
        seed=args.seed,
        quest_length=args.quest_length,
        nb_rooms=args.nb_rooms,
        nb_objects=args.nb_objects,
        max_episode_steps=args.max_steps,
    )

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
