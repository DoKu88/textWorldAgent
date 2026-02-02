"""Base agent class for TextWorld environments with Pydantic type guarantees."""

import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import torch
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Load environment variables from .env
load_dotenv()

# =============================================================================
# Pydantic Models for Agent I/O
# =============================================================================

class AgentInput(BaseModel):
    """Standardized input for all agents."""
    observation: str = Field(..., description="Current game observation/description")
    score: int = Field(default=0, description="Current game score")
    done: bool = Field(default=False, description="Whether the game has ended")
    admissible_commands: List[str] = Field(
        default_factory=lambda: ["look"],
        description="List of valid commands the agent can execute"
    )
    inventory: Optional[str] = Field(default=None, description="Current inventory")
    max_score: Optional[int] = Field(default=None, description="Maximum possible score")

    @classmethod
    def from_textworld(cls, observation: str, score: int, done: bool, info: dict) -> "AgentInput":
        """Factory method to create AgentInput from TextWorld env output."""
        return cls(
            observation=observation,
            score=score,
            done=done,
            admissible_commands=info.get("admissible_commands", ["look"]),
            inventory=info.get("inventory"),
            max_score=info.get("max_score"),
        )

class AgentOutput(BaseModel):
    """Standardized output from all agents."""
    action: str = Field(..., description="The chosen action to execute")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning for the action")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")

# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """Abstract base agent for TextWorld using Pydantic I/O."""

    name: str = "base"

    @abstractmethod
    def act(self, agent_input: AgentInput) -> AgentOutput:
        """Select an action given the current game state."""
        pass

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass

    def __call__(self, observation: str, score: int, done: bool, info: dict) -> str:
        """Convenience method for direct TextWorld integration."""
        agent_input = AgentInput.from_textworld(observation, score, done, info)
        agent_output = self.act(agent_input)
        return agent_output.action

# =============================================================================
# Agent Implementations
# =============================================================================

class RandomAgent(BaseAgent):
    """Agent that selects actions randomly from admissible commands."""

    name: str = "random"

    def act(self, agent_input: AgentInput) -> AgentOutput:
        action = random.choice(agent_input.admissible_commands)
        return AgentOutput(
            action=action,
            reasoning="Randomly selected from admissible commands",
            confidence=1.0 / len(agent_input.admissible_commands)
        )


class LLMAgentTransformers(BaseAgent):
    """Agent that uses a local HuggingFace Transformers LLM to select actions."""

    name: str = "llm-transformers"

    def __init__(self, model_name: str = "google/flan-t5-small", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.history: List[str] = []

        if self.verbose:
            print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use seq2seq for T5/Flan models, causal for others
        if "t5" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
            if self.verbose:
                print("Using seq2seq mode")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False
            if self.verbose:
                print("Using causal mode")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset(self) -> None:
        self.history = []

    def _clean_observation(self, observation: str) -> str:
        """Remove ASCII art and special formatting from observation."""
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
        return " ".join(obs_clean.split())  # normalize whitespace

    def _build_prompt(self, agent_input: AgentInput) -> tuple[str, Dict[str, str]]:
        """Build prompt and options mapping."""
        options: Dict[str, str] = {
            str(i + 1): cmd for i, cmd in enumerate(agent_input.admissible_commands)
        }
        options_str = ", ".join(f"{k}. {v}" for k, v in options.items())
        obs_clean = self._clean_observation(agent_input.observation)

        prompt = f"You are in a text game. {obs_clean} Choose the best action: {options_str}"
        return prompt, options

    def _parse_response(self, response: str, options: Dict[str, str], admissible_commands: List[str]) -> Optional[str]:
        """Parse LLM response to extract chosen command."""
        # Try to parse as a numbered choice (e.g. "1", "2.", " 3 ")
        parts = response.strip().split()
        if parts:
            first = parts[0].rstrip(".")
            if first in options:
                return options[first]

        # Fallback: word overlap scoring
        best_cmd = None
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

        return best_cmd if best_score > 0 else None

    def act(self, agent_input: AgentInput) -> AgentOutput:
        prompt, options = self._build_prompt(agent_input)

        if self.verbose:
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

        if self.verbose:
            print(f"[RAW LLM OUTPUT]: '{response}'")
            print(f"Options: {options}")

        action = self._parse_response(response, options, agent_input.admissible_commands)

        if action is None:
            raise RuntimeError(
                f"LLM agent did not produce a valid admissible command.\n"
                f"Model response: '{response}'\n"
                f"Admissible commands: {agent_input.admissible_commands}"
            )

        if self.verbose:
            print(f"Chosen: {action}")

        return AgentOutput(
            action=action,
            reasoning=f"LLM response: {response}",
            confidence=None  # Could compute based on logits
        )


class AgentOpenAI(BaseAgent):
    """Agent that uses the OpenAI API to select actions."""

    name: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = True,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

        # Get API key from parameter, env var, or .env file
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)

        if self.verbose:
            print(f"Using OpenAI model: {model}")

    def reset(self) -> None:
        self.history = []

    def _clean_observation(self, observation: str) -> str:
        """Remove ASCII art and special formatting from observation."""
        lines = observation.split('\n')
        clean_lines = []
        for line in lines:
            special_chars = sum(1 for c in line if c in '$\\|_/>()[]{}')
            if len(line) > 0 and special_chars / len(line) > 0.15:
                continue
            if '$$' in line or '\\$' in line:
                continue
            line = line.replace("-=", "").replace("=-", "").strip()
            if line:
                clean_lines.append(line)

        obs_clean = " ".join(clean_lines)
        return " ".join(obs_clean.split())

    def act(self, agent_input: AgentInput) -> AgentOutput:
        obs_clean = self._clean_observation(agent_input.observation)

        # Build the prompt
        commands_list = "\n".join(
            f"- {cmd}" for cmd in agent_input.admissible_commands
        )

        system_prompt = """You are an expert text adventure game player. Your goal is to complete the game objectives efficiently.

Rules:
1. You MUST respond with EXACTLY one of the available commands, nothing else.
2. Do not add any explanation or extra text.
3. Think about what action will make progress toward the game goal."""

        user_prompt = f"""Current situation: {obs_clean}

Available commands:
{commands_list}

Respond with exactly one command from the list above:"""

        if self.verbose:
            print(f"\n[PROMPT]: {user_prompt}\n")

        # Call OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=50,
            temperature=0.0,
        )

        raw_response = response.choices[0].message.content.strip()

        if self.verbose:
            print(f"[RAW LLM OUTPUT]: '{raw_response}'")

        # Match response to admissible commands
        action = None

        # Try exact match first
        for cmd in agent_input.admissible_commands:
            if cmd.lower() == raw_response.lower():
                action = cmd
                break

        # Try substring match
        if action is None:
            for cmd in agent_input.admissible_commands:
                if cmd.lower() in raw_response.lower():
                    action = cmd
                    break

        # Try word overlap as last resort
        if action is None:
            best_cmd = None
            best_score = 0
            response_words = set(raw_response.lower().split())

            for cmd in agent_input.admissible_commands:
                cmd_words = set(cmd.lower().split())
                overlap = len(response_words & cmd_words)
                if overlap > best_score:
                    best_score = overlap
                    best_cmd = cmd

            if best_score > 0:
                action = best_cmd

        if action is None:
            raise RuntimeError(
                f"OpenAI agent did not produce a valid admissible command.\n"
                f"Model response: '{raw_response}'\n"
                f"Admissible commands: {agent_input.admissible_commands}"
            )

        if self.verbose:
            print(f"Chosen: {action}")

        return AgentOutput(
            action=action,
            reasoning=raw_response,
            confidence=None,
        )


# =============================================================================
# Agent Factory
# =============================================================================

class AgentFactory:
    """Factory for creating agent instances."""

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class with a name."""
        cls._registry[name] = agent_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAgent:
        """Create an agent instance by name."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown agent: {name}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
        return list(cls._registry.keys())


# Register built-in agents
AgentFactory.register("random", RandomAgent)
AgentFactory.register("llm", LLMAgentTransformers)
AgentFactory.register("llm-transformers", LLMAgentTransformers)
AgentFactory.register("openai", AgentOpenAI)
