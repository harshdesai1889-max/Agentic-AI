# ============================================================
# DAY223 AGENTS - SAFE SOC REFACTOR
# Run this from the root of your day223_agents project.
# Creates a timestamped backup before changing any files.
# Does NOT read, overwrite, or print your .env file.
# ============================================================

$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path

if (!(Test-Path (Join-Path $Root "main.py")) -or
    !(Test-Path (Join-Path $Root "config\settings.py"))) {
    throw "Run this script from the day223_agents project root. main.py and config\settings.py must exist."
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Backup = Join-Path $Root "backup_before_soc_$Timestamp"
New-Item -ItemType Directory -Path $Backup -Force | Out-Null

$BackupTargets = @(
    "main.py",
    "test.py",
    "requirements.txt",
    ".gitignore",
    "agents",
    "config",
    "core",
    "prompts",
    "tools"
)

foreach ($Target in $BackupTargets) {
    $Source = Join-Path $Root $Target
    if (Test-Path $Source) {
        Copy-Item -LiteralPath $Source -Destination $Backup -Recurse -Force
    }
}

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-ProjectFile {
    param(
        [Parameter(Mandatory=$true)][string]$RelativePath,
        [Parameter(Mandatory=$true)][string]$Content
    )

    $FullPath = Join-Path $Root $RelativePath
    $Directory = Split-Path -Parent $FullPath

    if (!(Test-Path $Directory)) {
        New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    }

    [System.IO.File]::WriteAllText($FullPath, $Content, $Utf8NoBom)
    Write-Host "Updated: $RelativePath" -ForegroundColor DarkGreen
}

Write-Host "`nBackup created: $Backup" -ForegroundColor Cyan
Write-Host "Writing modular project files..." -ForegroundColor Cyan

# ------------------------------------------------------------
# PACKAGE MARKERS
# ------------------------------------------------------------

$EmptyInit = @'
"""Package marker."""
'@

Write-ProjectFile "config\__init__.py" $EmptyInit
Write-ProjectFile "core\__init__.py" $EmptyInit
Write-ProjectFile "agents\__init__.py" $EmptyInit
Write-ProjectFile "tools\__init__.py" $EmptyInit
Write-ProjectFile "prompts\__init__.py" $EmptyInit
Write-ProjectFile "utils\__init__.py" $EmptyInit
Write-ProjectFile "tests\__init__.py" $EmptyInit

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

Write-ProjectFile "config\settings.py" @'
import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Add it to your .env file."
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Preserve the two NVIDIA Nemotron models used by this project.
MODEL1 = "nvidia/nemotron-3.5-lightning:free"
MODEL2 = "nvidia/nemotron-3-ultra-550b-a55b:free"

MAX_TOOL_ROUNDS = 5
MAX_PLANNED_TASKS = 10
'@

# ------------------------------------------------------------
# SHARED OPENROUTER CLIENT
# ------------------------------------------------------------

Write-ProjectFile "core\llm.py" @'
from openai import OpenAI

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

# One shared OpenRouter client for the whole application.
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)
'@

# ------------------------------------------------------------
# MODEL ROUTER
# ------------------------------------------------------------

Write-ProjectFile "core\model_router.py" @'
from config.settings import MODEL1, MODEL2


def select_model(task_type: str) -> str:
    """
    Select a model using a simple explicit routing policy.

    "simple" / "fast" -> Nemotron Lightning
    "complex" / "advanced" -> Nemotron Ultra
    Unknown values default to Lightning.
    """
    normalized = task_type.strip().lower()

    if normalized in {"complex", "advanced", "reasoning", "planning"}:
        return MODEL2

    return MODEL1
'@

# ------------------------------------------------------------
# PROMPTS
# assistant.txt is plain text, NOT a Python assignment.
# ------------------------------------------------------------

Write-ProjectFile "prompts\assistant.txt" @'
You are an intelligent general-purpose AI assistant powered by NVIDIA Nemotron.

You have two major capabilities:
1. INTERNAL MODEL INTELLIGENCE
2. EXTERNAL TOOL EXECUTION

INTERNAL MODEL INTELLIGENCE

Use your pretrained knowledge and reasoning abilities to answer questions involving:
- Science and mathematics
- Programming and software engineering
- AI and machine learning
- Business and finance concepts
- Stock market analysis and investing concepts
- Education and research
- General knowledge
- Writing and problem solving
- Complex reasoning and explanations

Do not unnecessarily call tools when you can answer effectively using your own knowledge.
You are allowed and expected to answer general conceptual questions directly from your pretrained knowledge.

EXTERNAL TOOLS

Use available tools when the task requires:
- Accurate mathematical calculations
- Current date and time
- Word and character counting
- External data retrieval
- Live information, when suitable tools exist

TOOL RULES

1. Decide whether a tool is actually necessary.
2. Use your own reasoning whenever appropriate.
3. Never invent tool results.
4. Interpret tool results before answering.
5. Combine internal knowledge with external information when both are useful.
6. Clearly distinguish known facts from uncertain information.
7. A tool result is input for your reasoning, not automatically the complete answer.
8. Never claim that a tool was used when it was not.

GENERAL BEHAVIOUR

- Answer questions directly.
- Provide detailed explanations when requested.
- Analyse complex problems carefully.
- Never pretend to have live data without a live data source.
- Clearly communicate uncertainty.
- If a request needs current external information and no suitable tool exists, state that limitation instead of fabricating an answer.
'@

Write-ProjectFile "prompts\planner.txt" @'
You are an expert AI task planner.

Break the user's complex objective into a small number of logical, specific, ordered, achievable subtasks.

Return ONLY a valid JSON object in this exact general structure:
{
  "tasks": [
    {
      "id": 1,
      "description": "A specific task description"
    }
  ]
}

Rules:
- The top-level object must contain a "tasks" array.
- Each task must contain an integer "id" and a non-empty string "description".
- Keep tasks in a sensible execution order.
- Make each task independently executable where possible.
- Do not execute the tasks.
- Do not include Markdown fences or commentary outside the JSON.
- Do not create unnecessary tasks.
'@

Write-ProjectFile "prompts\executor.txt" @'
You are executing one assigned task as part of a larger objective.

Complete only the assigned subtask, while preserving relevant context supplied by the user.
Use your pretrained knowledge and reasoning. Use available tools when they materially improve accuracy.
Do not invent facts, tool results, sources, or completed actions.
Clearly identify assumptions, missing information, and uncertainty.
Return a useful, structured result that can be combined with results from other subtasks.
'@

Write-ProjectFile "prompts\synthesizer.txt" @'
You are an expert analytical report writer.

Combine the original objective and all task results into one accurate, coherent, structured final response.

Requirements:
- Address the original objective directly.
- Organize the response with useful headings.
- Remove unnecessary repetition.
- Preserve important findings, calculations, and distinctions.
- Do not invent information or claim that missing work was completed.
- Identify missing, conflicting, or uncertain information clearly.
- Distinguish conclusions supported by task results from assumptions.
'@

# ------------------------------------------------------------
# PROMPT LOADER
# ------------------------------------------------------------

Write-ProjectFile "utils\prompt_loader.py" @'
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


@lru_cache(maxsize=32)
def load_prompt(prompt_name: str) -> str:
    """
    Load a plain-text prompt from the project's prompts directory.
    Only simple filenames are accepted; directory traversal is rejected.
    """
    if not prompt_name or Path(prompt_name).name != prompt_name:
        raise ValueError("Invalid prompt filename.")

    prompt_path = PROMPTS_DIR / prompt_name

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt not found: {prompt_name}")

    content = prompt_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Prompt is empty: {prompt_name}")

    return content
'@

# ------------------------------------------------------------
# JSON UTILITY FOR MODEL OUTPUT
# ------------------------------------------------------------

Write-ProjectFile "utils\json_utils.py" @'
import json
from typing import Any


def parse_json_object(content: str) -> dict[str, Any]:
    """
    Parse a JSON object returned by an LLM.
    Accepts either raw JSON or a response surrounded by a Markdown fence.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned empty JSON content.")

    text = content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object.")

        parsed = json.loads(text[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object at the top level.")

    return parsed
'@

# ------------------------------------------------------------
# BASIC TOOLS - SAFE AST CALCULATOR, NO eval()
# ------------------------------------------------------------

Write-ProjectFile "tools\basic_tools.py" @'
import ast
import math
import operator
from datetime import datetime


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 50
_MAX_ABS_VALUE = 10 ** 100
_MAX_ABS_EXPONENT = 10


def _check_number(value):
    if type(value) not in (int, float):
        raise ValueError("Only real numeric values are supported.")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numeric results are not allowed.")

    if abs(value) > _MAX_ABS_VALUE:
        raise ValueError("Numeric value is too large.")

    return value


def calculator(expression: str):
    """
    Safely evaluate basic arithmetic using an AST whitelist.
    Names, function calls, attributes, containers and Python code are rejected.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "A non-empty mathematical expression is required."}

    expression = expression.strip()

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return {"error": "Expression is too long."}

    try:
        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("Expression is too complex.")

        def evaluate(node):
            if isinstance(node, ast.Constant):
                return _check_number(node.value)

            if isinstance(node, ast.BinOp):
                operation = _ALLOWED_OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Unsupported mathematical operator.")

                left = evaluate(node.left)
                right = evaluate(node.right)

                if isinstance(node.op, ast.Pow):
                    if abs(right) > _MAX_ABS_EXPONENT:
                        raise ValueError("Exponent is outside the permitted range.")

                result = operation(left, right)
                return _check_number(result)

            if isinstance(node, ast.UnaryOp):
                operation = _ALLOWED_OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Unsupported unary operator.")

                return _check_number(operation(evaluate(node.operand)))

            raise ValueError("Invalid or unsupported mathematical expression.")

        result = evaluate(tree.body)
        return {"result": result}

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
        return {"error": str(error)}

    except Exception:
        return {"error": "Calculation failed."}


def get_datetime():
    """Return the current server-local date and time, including timezone offset."""
    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


def word_counter(text: str):
    """Count whitespace-separated words and characters."""
    if not isinstance(text, str):
        return {"error": "The text argument must be a string."}

    return {
        "word_count": len(text.split()),
        "character_count": len(text),
        "characters_without_spaces": len("".join(text.split())),
    }
'@

# ------------------------------------------------------------
# TOOL SCHEMAS - WHAT THE MODEL SEES
# ------------------------------------------------------------

Write-ProjectFile "tools\schemas.py" @'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Calculate a basic mathematical expression accurately. "
                "Use for arithmetic instead of estimating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A basic arithmetic expression, for example (25 * 4) + 10.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time of the machine running this agent.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_counter",
            "description": "Count words, characters, and characters excluding whitespace in supplied text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyse.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
]
'@

# ------------------------------------------------------------
# TOOL REGISTRY - NAME TO PYTHON IMPLEMENTATION
# ------------------------------------------------------------

Write-ProjectFile "tools\registry.py" @'
from tools.basic_tools import (
    calculator,
    get_datetime,
    word_counter,
)

TOOL_REGISTRY = {
    "calculator": calculator,
    "get_datetime": get_datetime,
    "word_counter": word_counter,
}
'@

# ------------------------------------------------------------
# TOOL EXECUTOR - VALIDATES MODEL ARGUMENTS BEFORE RUNNING
# ------------------------------------------------------------

Write-ProjectFile "tools\executor.py" @'
import json
import logging

from tools.registry import TOOL_REGISTRY


logger = logging.getLogger(__name__)

_EXPECTED_ARGUMENTS = {
    "calculator": {"expression"},
    "get_datetime": set(),
    "word_counter": {"text"},
}


def execute_tool(tool_call) -> dict:
    """
    Execute a model-requested tool only if its name and arguments are valid.
    tool_call is an OpenAI-compatible ChatCompletionMessageToolCall object.
    """
    try:
        function = getattr(tool_call, "function", None)

        if function is None:
            return {"error": "Malformed tool call."}

        tool_name = getattr(function, "name", None)

        if tool_name not in TOOL_REGISTRY:
            return {"error": "Unknown tool requested."}

        raw_arguments = getattr(function, "arguments", "{}")

        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            return {"error": "Tool arguments must be a JSON object."}

        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be a JSON object."}

        expected = _EXPECTED_ARGUMENTS[tool_name]

        if set(arguments.keys()) != expected:
            return {
                "error": (
                    f"Invalid arguments for {tool_name}. "
                    f"Expected exactly: {sorted(expected)}"
                )
            }

        if tool_name == "calculator" and not isinstance(arguments["expression"], str):
            return {"error": "The expression argument must be a string."}

        if tool_name == "word_counter" and not isinstance(arguments["text"], str):
            return {"error": "The text argument must be a string."}

        result = TOOL_REGISTRY[tool_name](**arguments)

        if not isinstance(result, dict):
            return {"result": result}

        return result

    except json.JSONDecodeError:
        return {"error": "Tool arguments contained invalid JSON."}

    except Exception:
        logger.exception("Tool execution failed.")
        return {"error": "Tool execution failed. Check the application logs."}
'@

# ------------------------------------------------------------
# MAIN TOOL-CALLING AGENT ENGINE
# ------------------------------------------------------------

Write-ProjectFile "core\agent_engine.py" @'
import json
import logging

from config.settings import MAX_TOOL_ROUNDS, MODEL1
from core.llm import client
from tools.executor import execute_tool
from tools.schemas import TOOLS
from utils.prompt_loader import load_prompt


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = load_prompt("assistant.txt")


def run_agent(user_task: str, model: str = MODEL1) -> str:
    """
    Run a general-purpose Nemotron assistant.

    The model can answer using its pretrained knowledge or request tools.
    Tool results are returned to the model so it can interpret them and
    produce a natural-language answer.
    """
    if not isinstance(user_task, str) or not user_task.strip():
        raise ValueError("user_task must be a non-empty string.")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("A valid model identifier is required.")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_task.strip(),
        },
    ]

    for round_number in range(MAX_TOOL_ROUNDS):
        logger.info("Agent model request, tool round %s", round_number + 1)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        if not response.choices:
            raise RuntimeError("The model returned no response choices.")

        message = response.choices[0].message

        # No tool call: answer directly from model knowledge or conversation.
        if not message.tool_calls:
            return message.content or "The model returned an empty response."

        # Preserve the assistant tool-call message, including tool-call IDs.
        messages.append(message.model_dump(exclude_none=True))

        # Execute each requested tool and return its result to the model.
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    # The tool-round limit has been reached. Ask the model to finish using
    # the tool results already present, without allowing another tool call.
    messages.append(
        {
            "role": "user",
            "content": (
                "The tool execution limit has been reached. "
                "Now provide the best complete answer using the conversation "
                "and tool results already available. Do not request more tools."
            ),
        }
    )

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if not final_response.choices:
        raise RuntimeError("The model returned no final response.")

    return (
        final_response.choices[0].message.content
        or "The model returned an empty final response."
    )
'@

# ------------------------------------------------------------
# PLANNER - MODEL2 CREATES A VALIDATED TASK LIST
# ------------------------------------------------------------

Write-ProjectFile "agents\planner.py" @'
from config.settings import MAX_PLANNED_TASKS, MODEL2
from core.llm import client
from utils.json_utils import parse_json_object
from utils.prompt_loader import load_prompt


PLANNER_PROMPT = load_prompt("planner.txt")


def create_research_plan(objective: str) -> dict:
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("The planning objective must be a non-empty string.")

    response = client.chat.completions.create(
        model=MODEL2,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": objective.strip()},
        ],
        response_format={"type": "json_object"},
    )

    if not response.choices:
        raise RuntimeError("The planner returned no response choices.")

    content = response.choices[0].message.content
    plan = parse_json_object(content)

    tasks = plan.get("tasks")

    if not isinstance(tasks, list):
        raise ValueError("Planner output must contain a 'tasks' list.")

    if len(tasks) > MAX_PLANNED_TASKS:
        tasks = tasks[:MAX_PLANNED_TASKS]

    validated_tasks = []

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue

        description = task.get("description")

        if not isinstance(description, str) or not description.strip():
            continue

        task_id = task.get("id", index)

        if not isinstance(task_id, int) or isinstance(task_id, bool):
            task_id = index

        validated_tasks.append(
            {
                "id": task_id,
                "description": description.strip(),
            }
        )

    return {"tasks": validated_tasks}
'@

# ------------------------------------------------------------
# TASK EXECUTOR - USES THE SAME TOOL-ENABLED AGENT ENGINE
# ------------------------------------------------------------

Write-ProjectFile "agents\executor.py" @'
from config.settings import MODEL1
from core.agent_engine import run_agent
from utils.prompt_loader import load_prompt


EXECUTOR_PROMPT = load_prompt("executor.txt")


def execute_research_task(task: str) -> str:
    """
    Execute one planned task through the normal agent engine.
    This means research tasks retain model knowledge AND tool access.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Research task must be a non-empty string.")

    task_message = (
        f"{EXECUTOR_PROMPT}\n\n"
        "ASSIGNED SUBTASK:\n"
        f"{task.strip()}"
    )

    return run_agent(
        user_task=task_message,
        model=MODEL1,
    )
'@

# ------------------------------------------------------------
# RESEARCH ORCHESTRATOR - PLAN, EXECUTE, SYNTHESIZE
# ------------------------------------------------------------

Write-ProjectFile "agents\researcher.py" @'
import json
import logging

from config.settings import MODEL2
from core.llm import client
from agents.planner import create_research_plan
from agents.executor import execute_research_task
from utils.prompt_loader import load_prompt


logger = logging.getLogger(__name__)
SYNTHESIZER_PROMPT = load_prompt("synthesizer.txt")


def run_research_agent(objective: str) -> str:
    """
    Run the multi-step workflow:
    MODEL2 plans -> MODEL1 executes tool-enabled tasks -> MODEL2 synthesizes.
    """
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("The research objective must be a non-empty string.")

    objective = objective.strip()

    print("\n[PLANNER] MODEL2 creating execution plan...")

    plan = create_research_plan(objective)
    tasks = plan.get("tasks", [])

    if not tasks:
        return "No valid research tasks were generated."

    results = []

    print(f"\n[EXECUTION] {len(tasks)} task(s) generated.")

    for task in tasks:
        task_id = task["id"]
        description = task["description"]

        print(f"\n[EXECUTING TASK {task_id}]")
        print(description)

        try:
            result = execute_research_task(description)
            status = "completed"
        except Exception as error:
            logger.exception("Research task %s failed.", task_id)
            result = f"Task execution failed: {error}"
            status = "failed"

        results.append(
            {
                "task_id": task_id,
                "description": description,
                "status": status,
                "result": result,
            }
        )

        print(f"[TASK {task_id}] {status.upper()}")

    print("\n[FINAL SYNTHESIS] MODEL2 preparing the report...")

    synthesis_input = (
        "Original objective:\n"
        f"{objective}\n\n"
        "Execution plan and task results:\n"
        f"{json.dumps(results, ensure_ascii=False, indent=2)}"
    )

    final_response = client.chat.completions.create(
        model=MODEL2,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIZER_PROMPT,
            },
            {
                "role": "user",
                "content": synthesis_input,
            },
        ],
    )

    if not final_response.choices:
        raise RuntimeError("The synthesizer returned no response choices.")

    return (
        final_response.choices[0].message.content
        or "The synthesizer returned an empty response."
    )
'@

# ------------------------------------------------------------
# CLI ENTRY POINT - MAIN.PY ONLY COORDINATES THE APPLICATION
# ------------------------------------------------------------

Write-ProjectFile "main.py" @'
from agents.researcher import run_research_agent
from config.settings import MODEL1, MODEL2
from core.agent_engine import run_agent


def display_header():
    print("\n========================================")
    print("           NEMOTRON AI AGENT")
    print("========================================")
    print("Provider: OpenRouter")
    print("Model 1: NVIDIA Nemotron 3.5 Lightning")
    print("Model 2: NVIDIA Nemotron 3 Ultra")
    print("----------------------------------------")


def display_menu():
    print("\nSelect operation:")
    print("1. Normal AI Assistant")
    print("2. Multi-Step AI Agent")
    print("3. Exit")


def select_model():
    print("\nSelect model:")
    print("1. MODEL1 - Nemotron 3.5 Lightning")
    print("2. MODEL2 - Nemotron 3 Ultra")

    choice = input("\nModel: ").strip()

    if choice == "1":
        return MODEL1

    if choice == "2":
        return MODEL2

    print("Invalid model selection.")
    return None


def run_normal_assistant():
    model = select_model()

    if model is None:
        return None

    task = input("\nEnter your task: ").strip()

    if not task:
        print("Please enter a task.")
        return None

    print("\n[AGENT] Processing request...")

    return run_agent(
        user_task=task,
        model=model,
    )


def run_multi_step_agent():
    objective = input("\nEnter your complex assignment: ").strip()

    if not objective:
        print("Please enter an objective.")
        return None

    print("\n[AGENT] Starting multi-step execution...")
    print("[PLANNER] MODEL2 creates the plan.")
    print("[EXECUTOR] MODEL1 executes each task with tool access.")
    print("[SYNTHESIZER] MODEL2 prepares the final result.")

    return run_research_agent(objective=objective)


def display_result(result):
    if result is None:
        return

    print("\n========================================")
    print("                  RESULT")
    print("========================================\n")
    print(result)
    print("\n========================================")


def main():
    display_header()

    while True:
        display_menu()
        choice = input("\nChoice: ").strip()

        if choice == "1":
            try:
                display_result(run_normal_assistant())
            except Exception as error:
                print(f"\n[ERROR] Normal assistant failed: {error}")

        elif choice == "2":
            try:
                display_result(run_multi_step_agent())
            except Exception as error:
                print(f"\n[ERROR] Multi-step agent failed: {error}")

        elif choice == "3":
            print("\nShutting down Nemotron AI Agent.")
            break

        else:
            print("\nInvalid choice. Select 1, 2 or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
'@

# ------------------------------------------------------------
# TEST.PY - TEST THE SAME SHARED CLIENT AND SETTINGS
# ------------------------------------------------------------

Write-ProjectFile "test.py" @'
from config.settings import MODEL1
from core.llm import client


def test_openrouter_connection():
    response = client.chat.completions.create(
        model=MODEL1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise AI assistant. "
                    "Reply with a short confirmation."
                ),
            },
            {
                "role": "user",
                "content": "Confirm that the OpenRouter connection works.",
            },
        ],
        temperature=0.2,
        max_tokens=100,
    )

    if not response.choices:
        raise RuntimeError("The model returned no response choices.")

    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_openrouter_connection()
'@

# ------------------------------------------------------------
# UNIT TESTS - LOCAL TESTS, NO API CALLS REQUIRED
# ------------------------------------------------------------

Write-ProjectFile "tests\test_basic_tools.py" @'
import unittest

from tools.basic_tools import calculator, get_datetime, word_counter
from tools.executor import execute_tool


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)


class BasicToolsTests(unittest.TestCase):

    def test_calculator_addition(self):
        self.assertEqual(calculator("2 + 3")["result"], 5)

    def test_calculator_parentheses_and_power(self):
        self.assertEqual(calculator("(2 + 3) ** 2")["result"], 25)

    def test_calculator_rejects_python_code(self):
        result = calculator("__import__('os').system('whoami')")
        self.assertIn("error", result)

    def test_calculator_rejects_large_exponent(self):
        result = calculator("2 ** 1000")
        self.assertIn("error", result)

    def test_calculator_handles_division_by_zero(self):
        result = calculator("5 / 0")
        self.assertIn("error", result)

    def test_datetime_returns_expected_fields(self):
        result = get_datetime()
        self.assertIn("datetime", result)
        self.assertIn("date", result)
        self.assertIn("time", result)
        self.assertIn("timezone", result)

    def test_word_counter(self):
        result = word_counter("hello world")
        self.assertEqual(result["word_count"], 2)
        self.assertEqual(result["character_count"], 11)
        self.assertEqual(result["characters_without_spaces"], 10)

    def test_tool_executor_valid_call(self):
        call = FakeToolCall(
            "calculator",
            '{"expression": "10 * 5"}',
        )
        self.assertEqual(execute_tool(call)["result"], 50)

    def test_tool_executor_rejects_unknown_tool(self):
        call = FakeToolCall("unknown_tool", "{}")
        self.assertIn("error", execute_tool(call))

    def test_tool_executor_rejects_extra_arguments(self):
        call = FakeToolCall(
            "calculator",
            '{"expression": "2 + 2", "extra": "not allowed"}',
        )
        self.assertIn("error", execute_tool(call))


if __name__ == "__main__":
    unittest.main()
'@

# ------------------------------------------------------------
# REQUIREMENTS
# ------------------------------------------------------------

Write-ProjectFile "requirements.txt" @'
openai>=1.55.0,<3.0.0
python-dotenv>=1.0.1,<2.0.0
'@

# ------------------------------------------------------------
# ENV EXAMPLE - REAL .env IS LEFT UNTOUCHED
# ------------------------------------------------------------

Write-ProjectFile ".env.example" @'
OPENROUTER_API_KEY=replace_with_your_openrouter_api_key
'@

# ------------------------------------------------------------
# GITIGNORE
# ------------------------------------------------------------

Write-ProjectFile ".gitignore" @'
# Secrets
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/
env/

# IDEs
.vscode/
.idea/

# Logs
*.log

# Backups created by the refactor script
backup_before_soc_*/

# Build artifacts
build/
dist/
*.egg-info/
'@

# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

Write-Host "`nSOC refactor files written successfully." -ForegroundColor Green
Write-Host "Backup: $Backup" -ForegroundColor Yellow
Write-Host "`nNext run these commands in this same terminal:" -ForegroundColor Cyan
Write-Host "  python -m pip install -r requirements.txt"
Write-Host "  python -m compileall -q ."
Write-Host "  python -m unittest discover -s tests -v"
Write-Host "  python main.py"
Write-Host "`nYour existing .env file was not changed." -ForegroundColor Green
