"""Environment wrapper for TextWorld games."""

import os
from typing import Optional, Tuple

import textworld
import textworld.gym
from textworld import GameMaker

def make_simple_game(game_path: str = "games/simple_game.z8") -> str:
    """Create a simple TextWorld game for testing."""

    os.makedirs("games", exist_ok=True)

    # Create a simple game using TextWorld's game maker
    game_maker = GameMaker()

    # Create rooms
    room = game_maker.new_room("Room")
    game_maker.set_player(room)

    # Add a simple object
    key = game_maker.new(type="k", name="key")
    room.add(key)

    # Add a container
    chest = game_maker.new(type="c", name="chest")
    chest.add_property("open")
    room.add(chest)

    # Create a simple quest: put the key in the chest
    game_maker.new_quest_using_commands(["take key", "put key in chest"])

    # Compile the game
    game_file = game_maker.compile(game_path)

    return game_file


def create_env(
    game_path: Optional[str] = None,
    request_infos: Optional[textworld.EnvInfos] = None,
    max_episode_steps: int = 100
):
    """Create a TextWorld gym environment."""

    if game_path is None:
        game_path = make_simple_game()

    if request_infos is None:
        request_infos = textworld.EnvInfos(
            admissible_commands=True,
            description=True,
            inventory=True,
            max_score=True,
            won=True,
            lost=True,
        )

    env_id = textworld.gym.register_game(
        game_path,
        request_infos=request_infos,
        max_episode_steps=max_episode_steps
    )

    env = textworld.gym.make(env_id)
    return env
