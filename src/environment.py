"""Environment wrapper for TextWorld games."""

import os
from typing import Optional

import textworld
import textworld.gym


def make_random_game(
    seed: Optional[int] = None,
    quest_length: int = 3,
    nb_rooms: int = 3,
    nb_objects: int = 5,
    games_dir: str = "games",
) -> str:
    """Create a random TextWorld game with a solvable quest.

    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
        quest_length: Number of actions required to complete the quest.
        nb_rooms: Number of rooms in the game world.
        nb_objects: Number of objects to place in the world.
        games_dir: Directory to save the compiled game file.

    Returns:
        Path to the compiled game file.
    """
    os.makedirs(games_dir, exist_ok=True)

    options = textworld.GameOptions()
    if seed is not None:
        options.seeds = seed
    options.nb_rooms = nb_rooms
    options.nb_objects = nb_objects
    options.quest_length = quest_length
    options.path = games_dir + "/"

    game_file, _ = textworld.make(options)
    return game_file


def create_env(
    game_path: Optional[str] = None,
    seed: Optional[int] = None,
    quest_length: int = 3,
    nb_rooms: int = 3,
    nb_objects: int = 5,
    request_infos: Optional[textworld.EnvInfos] = None,
    max_episode_steps: int = 100,
    intermediate_reward: bool = True,
    info_requests: Optional[dict] = None,
):
    """Create a TextWorld gym environment.

    Args:
        game_path: Path to existing game file. If None, generates a random game.
        seed: Random seed for game generation (only used if game_path is None).
        quest_length: Quest length for random game generation.
        nb_rooms: Number of rooms for random game generation.
        nb_objects: Number of objects for random game generation.
        request_infos: Information to request from the environment.
        max_episode_steps: Maximum steps before episode ends.
        intermediate_reward: Enable rewards for each sub-goal, not just final completion.
        info_requests: Dict of info flags to request from environment.

    Returns:
        TextWorld gym environment.
    """
    if game_path is None:
        game_path = make_random_game(
            seed=seed,
            quest_length=quest_length,
            nb_rooms=nb_rooms,
            nb_objects=nb_objects,
            games_dir="games",
        )

    if request_infos is None:
        # Use info_requests dict if provided, otherwise use defaults
        if info_requests is not None:
            request_infos = textworld.EnvInfos(
                admissible_commands=info_requests.get("admissible_commands", True),
                description=info_requests.get("description", True),
                inventory=info_requests.get("inventory", True),
                max_score=info_requests.get("max_score", True),
                won=info_requests.get("won", True),
                lost=info_requests.get("lost", True),
                objective=info_requests.get("objective", True),
                intermediate_reward=intermediate_reward,
            )
        else:
            request_infos = textworld.EnvInfos(
                admissible_commands=True,
                description=True,
                inventory=True,
                max_score=True,
                won=True,
                lost=True,
                objective=True,
                intermediate_reward=intermediate_reward,
            )

    env_id = textworld.gym.register_game(
        game_path,
        request_infos=request_infos,
        max_episode_steps=max_episode_steps,
    )

    env = textworld.gym.make(env_id)
    return env
