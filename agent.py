"""Math agent that solves questions using tools in a ReAct loop."""

import json

from dotenv import load_dotenv
from pydantic_ai import Agent
from calculator import calculate

load_dotenv()

# Configure your model below. Examples:
#   "google-gla:gemini-2.5-flash"       (needs GOOGLE_API_KEY)
#   "openai:gpt-4o-mini"                (needs OPENAI_API_KEY)
#   "anthropic:claude-sonnet-4-6"    (needs ANTHROPIC_API_KEY)
MODEL = "groq:llama-3.3-70b-versatile"

agent = Agent(
    MODEL,
    system_prompt=(
    "You are a strict agent.\n"
    "You MUST ALWAYS call tools.\n"
    "For product questions, ALWAYS call product_lookup.\n"
    "For math, ALWAYS call calculator_tool.\n"
    "DO NOT output function text.\n"
    "DO NOT answer directly.\n"
    )，
)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    with open("products.json", "r") as f:
        products = json.load(f)

    for name, price in products.items():
        if name.lower() == product_name.lower():
            return str(price)

    return ", ".join(products.keys())
# TODO: Implement this tool by uncommenting the code below and replacing
# the ... with your implementation. The tool should:
#   1. Read products.json using json.load() (json is already imported above)
#   2. If the product_name is in the catalog, return its price as a string
#   3. If not found, return the list of available product names so the agent
#      can try again with the correct name
#
# @agent.tool_plain
# def product_lookup(product_name: str) -> str:
#     """Look up the price of a product by name.
#     Use this when a question asks about product prices from the catalog.
#     """
#     ...


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = agent.run_sync(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()
