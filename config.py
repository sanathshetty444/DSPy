import os
from types import SimpleNamespace
import dspy
from dotenv import load_dotenv

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

# 1. Setup the Language Model (LM)
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key)

class MockRM:
    def __call__(self, query: str, k: int = 1, **kwargs):
        # A tiny local dictionary to act as our "database"
        mock_data = {
            "What castle did David Gregory inherit?": "David Gregory inherited Kinnairdy Castle in 1664.",
            "What is the capital of France?": "Paris is the capital of France.",
            "Who wrote Romeo and Juliet?": "William Shakespeare wrote Romeo and Juliet.",
            "What is the boiling point of water in Celsius?": "Water boils at 100 degrees Celsius."
        }
        
        # Get the text based on the query
        text = mock_data.get(query, "No relevant context found.")
        
        # Return a list containing an object with a .long_text attribute
        return [SimpleNamespace(long_text=text)]
# 3. Configure DSPy to use the MockRM
rm = MockRM()
lm.inspect_history(n=1)
# 3. Configure DSPy to use these globally
dspy.configure(lm=lm, rm=rm)