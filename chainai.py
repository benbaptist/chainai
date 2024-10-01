import litellm
import json

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class Step:
    prompt: str
    output: Optional[str] = None
    json: bool = False

class Model:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content

class Chain:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def add(self, prompt: str, output: Optional[str] = None, json: bool = False) -> 'Chain':
        self.steps.append({
            "prompt": prompt,
            "output": output,
            "json": json
        })
        return self

    def invoke(self, model: Model, verbose: bool = False) -> 'ChainOutput':
        for step in self.steps:
            prompt = step["prompt"]
            if isinstance(prompt, Chain):
                # Execute nested chain
                nested_output = prompt.invoke(model, verbose=verbose)
                self.context.update(nested_output.__dict__)
            else:
                # Construct the full prompt with all previous context
                full_prompt = "\n\n".join([
                    f"Previous context:",
                    "\n".join([f"{key}: {value}" for key, value in self.context.items()]),
                    f"Current task: {prompt}"
                ])

                # Replace placeholders in the prompt with context values
                for key, value in self.context.items():
                    full_prompt = full_prompt.replace(f"{{{key}}}", str(value))

                result = model.generate(full_prompt, json_mode=step["json"])

                if verbose:
                    print(f"Prompt: {full_prompt}")
                    print(f"Result: {result}")

                if step["json"]:
                    result = json.loads(result)

                if step["output"]:
                    self.context[step["output"]] = result
        return ChainOutput(self.context)

class ChainOutput:
    def __init__(self, context: Dict[str, Any]):
        self.__dict__.update(context)
