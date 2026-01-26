from google import genai
import os
import evaluate_LLMs as ev

ev.Get_prompts_for_LLM(amount = 5)

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='Tell me what NLI is'
)

print(response.text)

print(response.model_dump_json(
    exclude_none=True, indent=4))