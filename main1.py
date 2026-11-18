import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.basic_tools import (
    calculator,
    get_datetime,
    word_counter
)


# =====================================
# 1. ENVIRONMENT
# =====================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY missing from .env")


# =====================================
# 2. OPENROUTER CLIENT
# =====================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# =====================================
# 3. MODEL CONFIGURATION
# =====================================

MODEL1 = "nvidia/nemotron-3.5-lightning:free"

MODEL2 = "nvidia/nemotron-3-ultra-550b-a55b:free"


# =====================================
# 4. TOOL DEFINITIONS
# =====================================

TOOLS = [

    {
        "type": "function",

        "function": {
            "name": "calculator",

            "description": (
                "Calculate mathematical expressions. "
                "Use this for accurate arithmetic."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression"
                    }
                },

                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",

        "function": {
            "name": "get_datetime",

            "description": (
                "Get the current date and time."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",

        "function": {
            "name": "word_counter",

            "description": (
                "Count words and characters in text."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyse"
                    }
                },

                "required": ["text"]
            }
        }
    }

]


# =====================================
# 5. TOOL REGISTRY
# =====================================

TOOL_REGISTRY = {

    "calculator": calculator,

    "get_datetime": get_datetime,

    "word_counter": word_counter

}


# =====================================
# 6. SYSTEM INSTRUCTIONS
# =====================================

SYSTEM_PROMPT = """
You are an intelligent general-purpose AI assistant
powered by NVIDIA Nemotron.

You have two major capabilities:

1. INTERNAL MODEL INTELLIGENCE
2. EXTERNAL TOOL EXECUTION

INTERNAL MODEL INTELLIGENCE:

Use your pretrained knowledge and reasoning abilities
to answer questions involving:

- Science and mathematics
- Programming and software engineering
- AI and machine learning
- Business and finance concepts
- Stock market analysis and investing concepts
- Education and research
- General knowledge
- Writing and problem solving
- Complex reasoning and explanations

Do not unnecessarily call tools when you can
answer effectively using your own knowledge.

EXTERNAL TOOLS:

Use available tools when the task requires:

- Accurate mathematical calculations
- Current date and time
- Word and character counting
- External data retrieval
- Live information, when suitable tools exist

TOOL RULES:

1. Decide whether a tool is actually necessary.
2. Use your own reasoning whenever appropriate.
3. Never invent tool results.
4. Interpret tool results before answering.
5. Combine internal knowledge with external information
   when both are useful.
6. Clearly distinguish known facts from uncertain information.

GENERAL BEHAVIOUR:

- Answer questions directly.
- Provide detailed explanations when requested.
- Analyse complex problems carefully.
- Never pretend to have live data without a live data source.
- Clearly communicate uncertainty.
"""

# =====================================
# 7. MODEL SELECTION
# =====================================

def select_model(mode):

    if mode == "fast":
        return MODEL1

    elif mode == "advanced":
        return MODEL2

    else:
        return MODEL1


# =====================================
# 8. TOOL EXECUTION
# =====================================

def execute_tool(tool_call):

    tool_name = tool_call.function.name

    try:

        arguments = json.loads(
            tool_call.function.arguments
        )

        if tool_name not in TOOL_REGISTRY:

            return {
                "error": "Unknown tool requested"
            }

        if not isinstance(arguments, dict):

            return {
                "error": "Invalid tool arguments"
            }

        result = TOOL_REGISTRY[tool_name](**arguments)

        return result

    except json.JSONDecodeError:

        return {
            "error": "Invalid JSON arguments"
        }

    except TypeError:

        return {
            "error": "Incorrect tool parameters"
        }

    except Exception:

        return {
            "error": "Tool execution failed"
        }

# =====================================
# 9. MAIN AGENT ENGINE
# =====================================

def run_agent(user_task, model=MODEL1):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_task
        }

    ]

    MAX_TOOL_ROUNDS = 5

    for round_number in range(MAX_TOOL_ROUNDS):

        response = client.chat.completions.create(

            model=model,

            messages=messages,

            tools=TOOLS,

            tool_choice="auto"

        )

        message = response.choices[0].message

        # ---------------------------------
        # CASE 1: NO TOOL REQUIRED
        # ---------------------------------

        if not message.tool_calls:

            return message.content or "No response generated."

        # ---------------------------------
        # CASE 2: TOOL CALL REQUESTED
        # ---------------------------------

        messages.append(
            message.model_dump(exclude_none=True)
        )

        for tool_call in message.tool_calls:

            result = execute_tool(tool_call)

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": json.dumps(result)

            })

    # ---------------------------------
    # FINAL RESPONSE AFTER TOOL EXECUTION
    # ---------------------------------

    final_response = client.chat.completions.create(

        model=model,

        messages=messages

    )

    return final_response.choices[0].message.content

# =====================================
# 10. TASK PLANNER 
# =====================================
def create_plan(objective):

    response = client.chat.completions.create(
        model=MODEL2,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert AI task planner.

                Break complex objectives into smaller,
                logical and executable tasks.

                Return ONLY valid JSON in this format:

                {
                    "tasks": [
                        {
                            "id": 1,
                            "description": "Task description"
                        },
                        {
                            "id": 2,
                            "description": "Task description"
                        }
                    ]
                }

                Tasks must be specific, ordered and achievable.
                Do not execute the tasks.
                """
            },
            {
                "role": "user",
                "content": objective
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# =====================================
# 11. Multi-step Executor
# =====================================
def run_multi_step_agent(objective):

    print("\n[PLANNER] MODEL2 creating execution plan...")

    plan = create_plan(objective)

    tasks = plan.get("tasks", [])

    if not tasks:
        return "No executable tasks were generated."

    results = []

    print("\n[EXECUTION STARTED]")

    for task in tasks:

        task_description = task["description"]

        print(f"\n[EXECUTING TASK {task['id']}]")
        print(task_description)

        result = run_agent(
            user_task=task_description,
            model=MODEL1
        )

        results.append({
            "task_id": task["id"],
            "description": task_description,
            "result": result
        })

        print("[TASK COMPLETED]")

    print("\n[FINAL ANALYSIS] MODEL2")

    final_response = client.chat.completions.create(
        model=MODEL2,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert analytical AI.

                Combine the results of all completed tasks
                into one accurate, structured final response.

                Do not invent information.
                Identify missing information clearly.
                """
            },
            {
                "role": "user",
                "content": (
                    "Original objective:\n"
                    + objective
                    + "\n\nTask results:\n"
                    + json.dumps(results, indent=2)
                )
            }
        ]
    )

    return final_response.choices[0].message.content


# =====================================
# 12. APPLICATION
# =====================================

if __name__ == "__main__":

    print("\n================================")
    print("       NEMOTRON AI AGENT")
    print("================================")

    while True:

        print("\nSelect operation:")
        print("1. Normal AI Assistant")
        print("2. Multi-Step Agent")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "3":
            break

        if choice == "1":

            print("\nSelect model:")
            print("1. MODEL1 - Lightning")
            print("2. MODEL2 - Ultra")

            model_choice = input("\nModel: ").strip()

            if model_choice == "1":
                selected_model = MODEL1

            elif model_choice == "2":
                selected_model = MODEL2

            else:
                print("Invalid model selection.")
                continue

            task = input("\nEnter your task: ")

            result = run_agent(
                task,
                model=selected_model
            )

        elif choice == "2":

            task = input("\nEnter your complex assignment: ")

            result = run_multi_step_agent(task)

        else:

            print("Invalid selection.")
            continue

        print("\n================================")
        print("             RESULT")
        print("================================")

        print(result)