# ChainAI Project

## Overview
ChainAI is a Python module for creating and executing chains of AI tasks. It supports various AI models through the `litellm` backend, including large language models (LLMs). The module provides a structure for organizing and managing multi-step AI workflows.

## Example Usage
Here’s a brief example of how to use ChainAI:

```python
from chainai import Model, Chain

model = Model(
    model="gpt-4o-mini"
)

# Create chains

main_chain = (Chain()
    .add("Write a short story about a cat.", key="draft1")
    .add("Make it 10x better.", key="draft2")
    .add("Turn it into a haiku.", key="haiku")
)

output = main_chain.invoke(model)

# Output results
print(output.draft1)
print(output.draft2)
print(output.draft3)
``` 

## Requirements
To run this project, you need to have the following installed:
- Python 3.7 or later
- litellm
- requests
- numpy

## Installation
To install this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/benbaptist/chainai.git
   ```
2. Install the package:
    ```bash
    pip install .
    ```
3. ???
4. Profit

## Contributing
Contributions are welcome! Please check the contributions guidelines for more information.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Acknowledge any resources or collaborators that were helpful during the development of this project.