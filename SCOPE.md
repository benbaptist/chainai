Simple Python module for creating chains to execute a complex AI workflow, either with LLMs or with other AI models, using litellm as a backend.

# Example Usage

```python
from scope import Model, Chain

model = Model(
    model="gpt-4o-mini",
    temperature=0.0,
)

# Create chains

images_chain = Chain()
images_chain.add("Take the script and generate a list of images that should be in the video")

main_chain = Chain()
main_chain.add("Generate a concept of a shortform video about kittens dancing.")

# Insert the images chain into the main chain, creating a 'fork' that will automatically execute the images chain at this time, while the main chain continues to run
# This is the equivalent of: images_chain.invoke(model)

main_chain.add(images_chain)

# Continue the main chain
main_chain.add("Write a script for the video.")
main_chain.add("""Turn the script into a JSON object with the following schema: 
{
    "lines": [
        {
            "character": str,
            "dialogue": str,
        }
    ]
}
""", json=True)

output = main_chain.invoke(model)

# Output is a JSON object
print(output)
```
