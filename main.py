"""Application entry point for the Day 223/224 Multi-Provider Agentic AI System."""

from config.settings import MODEL1, MODEL2, GEMINI_MODEL, validate_config
from core.agent_engine import run_agent
from core.model_router import get_route
from agents.researcher import run_multi_step_agent


def print_banner():
    """Print the application header."""
    print("\n================================")
    print("       NEMOTRON AI AGENT")
    print("================================")


def handle_normal_assistant():
    """Handle normal single-turn assistant interactions with model/provider selection."""
    print("\nSelect model / provider:")
    print("1. MODEL1 - Nemotron Lightning (OpenRouter - Fast & Routine)")
    print("2. MODEL2 - Nemotron Ultra (OpenRouter - Complex Reasoning)")
    print("3. Gemini 3.8 Flash (Google GenAI - Optional Provider)")

    model_choice = input("\nChoice: ").strip()

    try:
        if model_choice == "1":
            route = get_route(mode="fast")
        elif model_choice == "2":
            route = get_route(mode="advanced")
        elif model_choice == "3":
            route = get_route(mode="gemini")
        else:
            print("Invalid model selection. Defaulting to MODEL1 (Nemotron Lightning).")
            route = get_route(mode="fast")
    except Exception as exc:
        print(f"\n[ROUTING ERROR] {str(exc)}")
        return

    task = input("\nEnter your task: ").strip()
    if not task:
        print("Task cannot be empty.")
        return

    result = run_agent(
        task,
        model=route.model,
        provider=route.provider,
    )
    display_result(result)


def handle_multi_step_agent():
    """Handle complex multi-step planning and research assignments."""
    task = input("\nEnter your complex assignment: ").strip()
    if not task:
        print("Assignment cannot be empty.")
        return

    result = run_multi_step_agent(task)
    display_result(result)


def display_result(result: str):
    """Format and print the execution output."""
    print("\n================================")
    print("             RESULT")
    print("================================")
    print(result)


def main():
    """Main CLI loop for the agent application."""
    try:
        validate_config()
    except RuntimeError as exc:
        print(f"\n[CONFIGURATION ERROR] {str(exc)}")
        print("Please check your .env file and ensure OPENROUTER_API_KEY is configured.")
        return

    print_banner()

    while True:
        print("\nSelect operation:")
        print("1. Normal AI Assistant")
        print("2. Multi-Step Agent")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "3":
            print("\nExiting Nemotron AI Agent. Goodbye!")
            break
        elif choice == "1":
            handle_normal_assistant()
        elif choice == "2":
            handle_multi_step_agent()
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
