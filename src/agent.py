"""Base agent class for TextWorld environments."""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

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

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use seq2seq for T5/Flan models, causal for others
        if "t5" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
            print("Using seq2seq mode")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False
            print("Using causal mode")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.history: List[str] = []

    def reset(self) -> None:
        self.history = []

    def _build_prompt(self, observation: str, admissible_commands: List[str]) -> Tuple[str, Dict[str, str]]:
        # Build options dictionary (number -> command)
        options: Dict[str, str] = {str(i + 1): cmd for i, cmd in enumerate(admissible_commands)}
        options_str = ", ".join(f"{k}. {v}" for k, v in options.items())

        # Clean observation: remove ASCII art and special formatting
        lines = observation.split('\n')
        clean_lines = []
        for line in lines:
            # Skip lines that look like ASCII art
            special_chars = sum(1 for c in line if c in '$\\|_/>()[]{}')
            if len(line) > 0 and special_chars / len(line) > 0.15:
                continue
            # Skip lines with $$ patterns (ASCII art)
            if '$$' in line or '\\$' in line:
                continue
            # Skip lines with -= formatting
            line = line.replace("-=", "").replace("=-", "").strip()
            if line:
                clean_lines.append(line)

        obs_clean = " ".join(clean_lines)
        obs_clean = " ".join(obs_clean.split())  # normalize whitespace

        prompt = f"You are in a text game. {obs_clean} Choose the best action: {options_str}"
        return prompt, options

    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        admissible_commands = info.get("admissible_commands", ["look"])

        prompt, options = self._build_prompt(observation, admissible_commands)
        print(f"\n[PROMPT]: {prompt}\n")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                input_length = inputs["input_ids"].shape[1]
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                new_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        response = response.strip().split("\n")[0].strip()
        print(f"[RAW LLM OUTPUT]: '{response}'")

        # Resolve chosen option: try number first, then fall back to word-overlap
        chosen_num = None
        best_cmd = None
        best_score = -1

        # Try to parse as a numbered choice (e.g. "1", "2.", " 3 ")
        parts = response.strip().split()
        if parts:
            first = parts[0].rstrip(".")
            if first in options:
                chosen_num = first
                best_cmd = options[chosen_num]

        if best_cmd is None:
            # Fallback: word overlap scoring
            best_score = -1
            response_words = set(response.lower().split())
            for cmd in admissible_commands:
                cmd_words = set(cmd.lower().split())
                overlap = len(response_words & cmd_words)
                if cmd.lower() == response.lower():
                    overlap += 10
                elif cmd.lower() in response.lower():
                    overlap += 5
                if overlap > best_score:
                    best_score = overlap
                    best_cmd = cmd

        print("Options:", options)
        if chosen_num is not None:
            print(f"Chosen: {chosen_num} -> {best_cmd}")
        else:
            print(f"Chosen (via word overlap): {best_cmd}")

        if best_cmd and (chosen_num is not None or best_score > 0):
            return best_cmd

        raise RuntimeError(
            f"LLM agent did not produce a valid admissible command.\n"
            f"Model response: '{response}'\n"
            f"Admissible commands: {admissible_commands}"
        )
