import litellm
import json

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class Step:
    """
    Represents a single step in a chain of prompts.

    Attributes:
        prompt (str): The prompt for this step.
        json (bool): Whether to request a JSON-formatted response.
        previous_output (Optional[str]): The output from the previous step.
        key (Optional[str]): An optional key for storing the result of this step.
    """
    prompt: str
    json: bool = False
    previous_output: Optional[str] = None
    key: Optional[str] = None

class Model:
    """
    Represents a language model used for generating responses.

    Attributes:
        model (str): The name or identifier of the language model.
        temperature (float): The temperature setting for response generation.
        kwargs (dict): Additional keyword arguments for model configuration, passed directly to litellm.completion
    """
    def __init__(self, model: str, temperature: float = 0.7, **kwargs):        
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def generate(self, prompt: str, json_mode: bool = False, **kwargs) -> str:
        """
        Generate a response using the configured language model. Typically, you will not call this method directly, but rather through .invoke() on a Chain.

        Args:
            prompt (str): The input prompt for the model.
            json_mode (bool, optional): Whether to request a JSON-formatted response. Defaults to False.
            **kwargs: Additional keyword arguments to override model configuration.

        Returns:
            str: The generated response from the model.

        Raises:
            Any exceptions raised by litellm.completion() will be propagated.
        """

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
    """
    Represents a sequence of steps to be executed by a language model.

    The Chain class allows for the creation of a series of prompts that can be
    executed sequentially, with each step potentially using the output of the
    previous steps. It supports nested chains and provides a flexible way to
    structure complex language model interactions.

    Attributes:
        steps (List[Step]): A list of Step objects representing the sequence of prompts.
        last_output (Optional[str]): The output from the most recently executed step.
        outputs (Dict[str, Any]): A dictionary storing named outputs from steps.

    Methods:
        add(prompt: str, key: Optional[str] = None, json: bool = False) -> 'Chain':
            Adds a new step to the chain.
        
        invoke(model: Model, verbose: bool = False) -> 'ChainOutput':
            Executes the chain using the provided model and returns the result.
    """
    def __init__(self):
        self.steps: List[Step] = []
        self.last_output: Optional[str] = None
        self.outputs: Dict[str, Any] = {}

    def add(self, prompt: str, key: Optional[str] = None, json: bool = False) -> 'Chain':
        """
        Adds a new step to the chain.

        Args:
            prompt (str): The prompt for this step.
            key (Optional[str], optional): An optional key for storing the result of this step. Defaults to None.
            json (bool, optional): Whether to request a JSON-formatted response. Defaults to False.

        Returns:
            Chain: The current Chain object, allowing for method chaining.
        """
        self.steps.append(Step(prompt=prompt, key=key, json=json))
        return self

    def invoke(self, model: Model, verbose: bool = False) -> 'ChainOutput':
        """
        Executes the chain using the provided model and returns the result.

        Args:
            model (Model): The language model to use for execution.
            verbose (bool, optional): Whether to print the prompts and results. Defaults to False.

        Returns:
            ChainOutput: An object containing the final result and any stored outputs.
        """
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
    """
    Represents the output from a Chain.

    Attributes:
        result (Any): The final result from the chain.
        outputs (Dict[str, Any]): A dictionary storing named outputs from steps.
    """
    def __init__(self, result: Any, outputs: Dict[str, Any]):
        self.result = result
        self.outputs = outputs

    def __getattr__(self, name):
        if name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"'ChainOutput' object has no attribute '{name}'")
