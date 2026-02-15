from module import RAG
import config
import dspy
from dspy.teleprompt import BootstrapFewShot

uncompiled_rag = RAG()
query = "What castle did David Gregory inherit?"

# Run it
response = uncompiled_rag(question=query)

print(f"Question: {query}")
print(f"Answer: {response.answer}")
print(f"Context used: {response.context[0][:100]}...")

# 1. Create a dummy trainset (In reality, you'd want 20-50 examples)
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare").with_inputs("question"),
    dspy.Example(question="What is the boiling point of water in Celsius?", answer="100").with_inputs("question"),
]

# 2. Define a success metric (Did the model output the exact right answer?)
def exact_match_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# 3. Initialize the Optimizer
optimizer = BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=2, # How many examples DSPy generates itself
    max_labeled_demos=2       # How many examples it pulls from your trainset
)

# 4. Compile the program!
print("Compiling program... This will make several API calls.")
compiled_rag = optimizer.compile(student=uncompiled_rag, trainset=trainset)

# Test the optimized model
optimized_response = compiled_rag(question=query)
print(f"Reasoning: {response.reasoning}") # <-- Added this!
print(f"Optimized Answer: {optimized_response.answer}")

# Save your optimized program for later use (so you don't have to re-compile)
compiled_rag.save("optimized_rag_program.json", save_program=False)
