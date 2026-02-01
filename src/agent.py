"""Base agent class for TextWorld environments."""

import random
from abc import ABC, abstractmethod
from typing import List, Optional
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseAgent(ABC):
    """Abstract base agent for TextWorld."""

    @abstractmethod
    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        """Select an action given the current game state."""
        pass

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects actions randomly from admissible commands."""

    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        admissible_commands = info.get("admissible_commands", ["look"])
        return random.choice(admissible_commands)


class LLMAgent(BaseAgent):
    """Agent that uses an LLM to select actions."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.history: List[str] = []

    def reset(self) -> None:
        self.history = []

    def _build_prompt(self, observation: str, admissible_commands: List[str]) -> str:
        commands_str = ", ".join(admissible_commands)

        prompt = f"""You are playing a text adventure game. Based on the observation, choose the best action.

Observation: {observation}

Available actions: {commands_str}

Your action:"""
        return prompt

    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        admissible_commands = info.get("admissible_commands", ["look"])

        prompt = self._build_prompt(observation, admissible_commands)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip().split("\n")[0].strip()

        # Match response to admissible commands
        response_lower = response.lower()
        for cmd in admissible_commands:
            if cmd.lower() in response_lower or response_lower in cmd.lower():
                return cmd

        # If no match, raise error and exit
        raise RuntimeError("LLM agent did not produce a valid admissible command.\n"
                           f"Observation: {observation}\n"
                           f"Model response: '{response}'\n"
                           f"Admissible commands: {admissible_commands}")
        sys.exit(1)
