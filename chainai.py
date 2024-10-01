import litellm
import json

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class Step:
    prompt: str
    json: bool = False
    previous_output: Optional[str] = None
    key: Optional[str] = None

class Model:
    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def generate(self, prompt: str, json_mode: bool = False, **kwargs) -> str:
        completion_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"} if json_mode else None,
            **self.kwargs,
            **kwargs
        }
        response = litellm.completion(**completion_kwargs)
        return response.choices[0].message.content
    
class Chain:
    def __init__(self):
        self.steps: List[Step] = []
        self.last_output: Optional[str] = None
        self.outputs: Dict[str, Any] = {}

    def add(self, prompt: str, key: Optional[str] = None, json: bool = False) -> 'Chain':
        self.steps.append(Step(prompt=prompt, key=key, json=json))
        return self

    def invoke(self, model: Model, verbose: bool = False) -> 'ChainOutput':
        for step in self.steps:
            prompt = step.prompt
            if isinstance(prompt, Chain):
                # Execute nested chain
                nested_output = prompt.invoke(model, verbose=verbose)
                self.last_output = nested_output.result
                self.outputs.update(nested_output.outputs)
            else:
                # Construct the full prompt with the previous output
                full_prompt = f"Previous output:\n{self.last_output}\n\nCurrent task: {prompt}"

                # Generate the result
                result = model.generate(full_prompt, json_mode=step.json)

                if verbose:
                    print(f"Prompt: {full_prompt}")
                    print(f"Result: {result}")

                if step.json:
                    result = json.loads(result)

                self.last_output = result

                if step.key:
                    self.outputs[step.key] = result

        return ChainOutput(self.last_output, self.outputs)

class ChainOutput:
    def __init__(self, result: Any, outputs: Dict[str, Any]):
        self.result = result
        self.outputs = outputs

    def __getattr__(self, name):
        if name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"'ChainOutput' object has no attribute '{name}'")
