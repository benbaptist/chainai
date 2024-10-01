from chainai import Model, Chain

# model = Model(
#     model="gpt-4o-mini",
#     temperature=0.0,
# )

model = Model(
    model="ollama_chat/llama3.2:1b",
    api_base="http://192.168.250.110:11434",
)

# Create chains

images_chain = (Chain()
    .add("Take the script and generate a list of images that should be in the video, associated with the time they should appear in the video", key="images")
)

main_chain = (Chain()
    .add("Generate a concept of a shortform video about kittens dancing.")
    .add(images_chain)
    .add("Write a script for the video.", key="script")
    .add("""Turn the script into a JSON object with the following schema: 
    {
        "lines": [
            {
                "character": str,
                "dialogue": str,
            }
        ]
    }
    """, json=True, key="script_json")
)

output = main_chain.invoke(model, verbose=True)
output_images = output.images
output_script = output.script_json

# Output results
print(output_images)
print(output_script)
