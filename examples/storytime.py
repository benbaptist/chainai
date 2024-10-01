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
print("***** Draft 1: *****")
print(output.draft1)
print("***** Draft 2: *****")
print(output.draft2)
print("***** Haiku: *****")
print(output.haiku)