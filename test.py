"""Standalone connectivity test script for the Nemotron OpenRouter client."""

from config.settings import MODEL1
from core.llm import client


def run_test():
    response = client.chat.completions.create(
        model=MODEL1,
        messages=[
            {
                "role": "system",
                "content": "You are AB SMART AI, an intelligent industrial AI assistant.",
            },
            {
                "role": "user",
                "content": "Explain how predictive maintenance works in a factory.",
            },
        ],
        temperature=0.3,
        max_tokens=500,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    run_test()
