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

        if self.verbose:
            print(f"{'='*50}")
            print("GAME START")
            print(f"{'='*50}")
            print(self._clean_obs(obs))

        while not done:
            # Use __call__ for backwards compatibility with TextWorld interface
            action = self.agent(obs, total_score, done, info)

            trajectory.append({
                "observation": obs,
                "action": action,
                "info": info.copy(),
                "score": total_score
            })

            if self.verbose:
                print(f"> {action}")

            obs, score, done, info = self.env.step(action)
            total_score = score
            steps += 1

            if self.verbose:
                print(self._clean_obs(obs))
                print(f"Score: {total_score}  |  Steps: {steps}\n")

        # Record final state
        trajectory.append({
            "observation": obs,
            "action": None,
            "info": info.copy(),
            "score": total_score,
            "done": True
        })

        if self.verbose:
            print(f"{'='*50}")
            result = "YOU WON!" if info.get("won", False) else "GAME OVER"
            print(f"{result}")
            print(f"Score: {total_score}  |  Steps: {steps}")
            print(f"{'='*50}")

        return total_score, steps, trajectory

    def run_episodes(self, n_episodes: int) -> List[Tuple[float, int, List[Dict]]]:
        """Run multiple episodes and return results."""

        results = []

        for ep in tqdm(range(n_episodes), desc="Running episodes", disable=self.verbose):
            result = self.run_episode()
            results.append(result)

        return results
