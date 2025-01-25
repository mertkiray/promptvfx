import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prompt_llm(prompt: str, system_message: str, temperature: float = 0.7) -> str:
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=temperature,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content or ""