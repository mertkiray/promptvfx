import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prompt_4o(prompt: str, system_message: str, temperature: float = 0.7) -> str:
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=temperature,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content or ""


def prompt_o1(prompt: str) -> str:
    completion = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ],
    )

    return completion.choices[0].message.content or ""