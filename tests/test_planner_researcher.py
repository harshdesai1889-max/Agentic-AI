import json
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from agents.planner import create_plan, validate_plan
from agents.executor import execute_task
from agents.researcher import run_multi_step_agent
from config.settings import MODEL2, MODEL1


def _mock_llm_response(content: str):
    """Helper to mock an LLM response choice."""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_planner_valid_plan():
    """Verify that valid JSON produces a structured tasks list."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({
            "tasks": [
                {"id": 1, "description": "Research market size"},
                {"id": 2, "description": "Analyze key competitors"},
            ]
        })
    )

    plan = create_plan(
        objective="Analyze EV battery market",
        client_instance=mock_client,
    )

    assert len(plan["tasks"]) == 2
    assert plan["tasks"][0]["id"] == 1
    assert plan["tasks"][0]["description"] == "Research market size"
    assert plan["tasks"][1]["id"] == 2


def test_planner_invalid_json():
    """Verify that malformed JSON is handled gracefully without crashing."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(
        "Here is your plan: {tasks: broken json..."
    )

    plan = create_plan(
        objective="Analyze quantum computing",
        client_instance=mock_client,
    )

    assert plan["tasks"] == []
    assert "error" in plan
    assert "Invalid planner JSON" in plan["error"]


def test_planner_empty_plan():
    """Verify handling of empty tasks list."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_llm_response(
        json.dumps({"tasks": []})
    )

    plan = create_plan(
        objective="Do nothing",
        client_instance=mock_client,
    )

    assert plan["tasks"] == []


def test_planner_excessive_number_of_tasks():
    """Verify that plans with excessive tasks are capped at max_tasks."""
    raw_tasks = [{"id": i, "description": f"Subtask {i}"} for i in range(1, 30)]
    validated = validate_plan({"tasks": raw_tasks}, max_tasks=10)

    assert len(validated["tasks"]) == 10
    assert validated["tasks"][0]["id"] == 1
    assert validated["tasks"][9]["id"] == 10


def test_executor_uses_agent_engine():
    """Verify that execute_task invokes the tool-enabled agent engine."""
    with patch("agents.executor.run_agent") as mock_run_agent:
        mock_run_agent.return_value = "Task result from agent engine"

        res = execute_task(task="Compute metrics", model=MODEL1)

        assert res == "Task result from agent engine"
        mock_run_agent.assert_called_once()
        call_kwargs = mock_run_agent.call_args[1]
        assert call_kwargs["user_task"] == "Compute metrics"
        assert call_kwargs["model"] == MODEL1


def test_multi_step_agent_full_workflow():
    """Verify the entire multi-step workflow from plan generation to final synthesis."""
    mock_client = MagicMock()

    # Step 1: Planner returns 2 tasks
    # Step 2: Synthesizer returns final analysis
    planner_output = json.dumps({
        "tasks": [
            {"id": 1, "description": "Gather financial data"},
            {"id": 2, "description": "Calculate profit margin"},
        ]
    })
    synthesis_output = "# Final Comprehensive Financial Report\nProfit margins increased by 15%."

    # First call is planner (MODEL2), second is synthesizer (MODEL2)
    mock_client.chat.completions.create.side_effect = [
        _mock_llm_response(planner_output),
        _mock_llm_response(synthesis_output),
    ]

    with patch("agents.researcher.execute_task") as mock_execute:
        mock_execute.side_effect = [
            "Financial data collected: Revenue = 100M",
            "Profit margin calculated: 15%",
        ]

        result = run_multi_step_agent(
            objective="Analyze annual company performance",
            client_instance=mock_client,
            verbose=False,
        )

        assert result == synthesis_output
        assert mock_execute.call_count == 2
        # Check that synthesizer received both task outputs
        synth_call_args = mock_client.chat.completions.create.call_args_list[1]
        synth_messages = synth_call_args[1]["messages"]
        user_message_content = synth_messages[1]["content"]
        assert "Financial data collected: Revenue = 100M" in user_message_content
        assert "Profit margin calculated: 15%" in user_message_content
        assert "Analyze annual company performance" in user_message_content


def test_multi_step_agent_preserves_task_failures():
    """Verify that failed tasks are not fabricated and failures are visible to synthesizer."""
    mock_client = MagicMock()

    planner_output = json.dumps({
        "tasks": [
            {"id": 1, "description": "Fetch database records"},
        ]
    })
    synthesis_output = "Report: Task 1 failed due to connection error. Incomplete data."

    mock_client.chat.completions.create.side_effect = [
        _mock_llm_response(planner_output),
        _mock_llm_response(synthesis_output),
    ]

    with patch("agents.researcher.execute_task") as mock_execute:
        # Simulate execution failure
        mock_execute.side_effect = RuntimeError("Database unreachable")

        result = run_multi_step_agent(
            objective="Database audit",
            client_instance=mock_client,
            verbose=False,
        )

        assert result == synthesis_output
        # Verify the failure was passed to synthesizer
        synth_call = mock_client.chat.completions.create.call_args_list[1]
        user_msg = synth_call[1]["messages"][1]["content"]
        assert "Task execution failed: Database unreachable" in user_msg

