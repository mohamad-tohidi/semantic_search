from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

client = OpenAI()

res = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "fuck_you"
        }
        ],
        model="gemma-3"
)

print(res)