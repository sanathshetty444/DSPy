from dspy import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short, fact-based answers based on the context provided."""
    
    context: str = dspy.InputField(desc="Relevant facts retrieved from Wikipedia")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Often between 1 and 5 words")