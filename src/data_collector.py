"""Data collection utilities for training (SFT, RL, DPO)."""

import json
from pathlib import Path
from typing import List, Dict, Any


class TrajectoryCollector:
    """Collects and saves game trajectories for training."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: List[List[Dict]] = []

    def add_trajectory(self, trajectory: List[Dict]) -> None:
        """Add a trajectory from a completed episode."""
        self.trajectories.append(trajectory)

    def save_sft_format(self, filename: str = "sft_data.jsonl") -> None:
        """Save trajectories in SFT format (prompt-completion pairs)."""

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for traj in self.trajectories:
                for step in traj:
                    if step.get("action") is None:
                        continue

                    obs = step["observation"]
                    action = step["action"]
                    commands = step["info"].get("admissible_commands", [])

                    prompt = f"Observation: {obs}\nAvailable actions: {', '.join(commands)}\nAction:"
                    completion = f" {action}"

                    record = {
                        "prompt": prompt,
                        "completion": completion
                    }
                    f.write(json.dumps(record) + "\n")

        print(f"Saved SFT data to {filepath}")

    def save_rl_format(self, filename: str = "rl_data.jsonl") -> None:
        """Save trajectories in RL format (with intermediate rewards).

        Uses TextWorld's intermediate_reward signal which provides:
        - Positive reward for sub-goal completion (good actions)
        - Zero for neutral actions
        - Negative reward for bad actions (if game defines penalties)
        """

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for traj in self.trajectories:
                for step in traj:
                    if step.get("action") is None:
                        continue

                    # Use TextWorld's intermediate_reward (step-wise reward signal)
                    # This is the reward that resulted from taking the action
                    intermediate_reward = step.get("intermediate_reward", 0)

                    record = {
                        "observation": step["observation"],
                        "action": step["action"],
                        "admissible_commands": step["info"].get("admissible_commands", []),
                        "intermediate_reward": intermediate_reward,
                        "score": step["score"]
                    }
                    f.write(json.dumps(record) + "\n")

        print(f"Saved RL data to {filepath}")

    def save_dpo_format(
        self,
        good_trajectories: List[List[Dict]],
        bad_trajectories: List[List[Dict]],
        filename: str = "dpo_data.jsonl"
    ) -> None:
        """Save paired trajectories in DPO format (chosen vs rejected)."""

        filepath = self.output_dir / filename

        # Pair up steps from good and bad trajectories
        pairs = []

        for good_traj, bad_traj in zip(good_trajectories, bad_trajectories):
            for good_step, bad_step in zip(good_traj, bad_traj):
                if good_step.get("action") is None or bad_step.get("action") is None:
                    continue

                # Only create pair if observations match (same state)
                if good_step["observation"] == bad_step["observation"]:
                    obs = good_step["observation"]
                    commands = good_step["info"].get("admissible_commands", [])

                    prompt = f"Observation: {obs}\nAvailable actions: {', '.join(commands)}\nAction:"

                    record = {
                        "prompt": prompt,
                        "chosen": f" {good_step['action']}",
                        "rejected": f" {bad_step['action']}"
                    }
                    pairs.append(record)

        with open(filepath, "w") as f:
            for record in pairs:
                f.write(json.dumps(record) + "\n")

        print(f"Saved DPO data to {filepath}")

    def clear(self) -> None:
        """Clear collected trajectories."""
        self.trajectories = []
