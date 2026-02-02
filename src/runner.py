"""Game runner for TextWorld agents."""

import re
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.agent import BaseAgent

class GameRunner:
    """Runs TextWorld games with an agent."""

    def __init__(self, agent: BaseAgent, env, verbose: bool = True):
        self.agent = agent
        self.env = env
        self.verbose = verbose

    def _clean_obs(self, obs: str) -> str:
        """Clean observation for display: remove game's inline score so we only show score in our Score line."""
        lines = obs.strip().split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            # Strip trailing score/max (e.g. "0/4" from "-= Room =-0/4")
            line = re.sub(r'\s*\d+/\d+\s*$', '', line).strip()
            # Skip lines that are only the game's "Score:" or "Score: X/Y"
            if not line or re.match(r'^Score:\s*(\d+/\d+)?\s*$', line):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def run_episode(self) -> Tuple[float, int, List[Dict]]:
        """Run a single episode and return (total_score, steps, trajectory).
           An episode is one full playthrough of a single game from start to finish.
        """

        obs, info = self.env.reset()
        self.agent.reset()

        done = False
        total_score = 0
        steps = 0
        trajectory = []

        max_score = info.get("max_score", "?")

        if self.verbose:
            print(f"{'─'*60}")
            print(f"  Game Start")
            print(f"{'─'*60}")
            print(f"\n{self._clean_obs(obs)}\n")

        while not done:
            # Use __call__ for backwards compatibility with TextWorld interface
            action = self.agent(obs, total_score, done, info)

            # Store pre-action state for trajectory
            pre_action_obs = obs
            pre_action_info = info.copy()
            pre_action_score = total_score

            if self.verbose:
                print(f"  > {action}")

            obs, score, done, info = self.env.step(action)
            total_score = score
            steps += 1

            # Extract intermediate reward (step-wise reward for the action taken)
            # This is the reward RESULTING from the action, returned by TextWorld
            intermediate_reward = info.get("intermediate_reward", 0)

            # Record transition: (state, action) -> reward
            trajectory.append({
                "observation": pre_action_obs,  # observation agent used to decide
                "action": action,
                "info": pre_action_info,  # info available when making decision
                "score": pre_action_score,  # score before action
                "intermediate_reward": intermediate_reward,  # reward from this action
            })

            if self.verbose:
                print(f"\n{self._clean_obs(obs)}")
                reward_str = f"+{intermediate_reward}" if intermediate_reward >= 0 else str(intermediate_reward)
                print(f"  [Score: {total_score}/{max_score} | Step: {steps} | Reward: {reward_str}]\n")

        # Record final state
        trajectory.append({
            "observation": obs,
            "action": None,
            "info": info.copy(),
            "score": total_score,
            "done": True
        })

        if self.verbose:
            print(f"{'─'*60}")
            result = "Victory!" if info.get("won", False) else "Game Over"
            print(f"  {result}  |  Score: {total_score}/{max_score}  |  Steps: {steps}")
            print(f"{'─'*60}")

        return total_score, steps, trajectory

    def run_episodes(self, n_episodes: int) -> List[Tuple[float, int, List[Dict]]]:
        """Run multiple episodes and return results."""

        results = []

        for ep in tqdm(range(n_episodes), desc="Running episodes", disable=self.verbose):
            result = self.run_episode()
            results.append(result)

        return results
