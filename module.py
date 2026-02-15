from signature import GenerateAnswer
import dspy
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        # The retriever module
        self.retrieve = dspy.Retrieve(k=num_passages)
        # The generator module wrapped in ChainOfThought
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question: str):
        # 1. Retrieve relevant context based on the question
        context = self.retrieve(question).passages
        
        # 2. Pass context and question to the generator
        prediction = self.generate_answer(context=context, question=question)
        
        # 3. Return a DSPy Prediction object
        return dspy.Prediction(context=context, reasoning=prediction.reasoning, answer=prediction.answer)