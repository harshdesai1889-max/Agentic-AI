"""
tests/test_specialized_agents.py

Comprehensive tests for Day 224 Part 4:
- Specialized PlannerAgent, ExecutorAgent, and ResearcherAgent
- Common BaseAgent architecture (identity, initialization, provider resolution)
- Responsibility boundary enforcement
- Tool execution and error handling
- Invalid input safety and provider failure resilience
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.provider_interface import LLMProvider, LLMMessage, LLMResponse, ToolCallRequest
from agents.base_agent import BaseAgent
from agents.planner import PlannerAgent, validate_plan, create_plan
from agents.executor import ExecutorAgent, execute_task
from agents.researcher import ResearcherAgent, run_multi_step_agent
from config.settings import MODEL1, MODEL2


# ===========================================================================
# Mock Providers for Offline Testing
# ===========================================================================

class MockTestProvider(LLMProvider):
    """Deterministic mock provider for testing specialized agents."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses) if responses else []
        self.error = error
        self.call_history = []

    def generate(
        self,
        messages,
        model,
        tools=None,
        tool_choice="auto",
    ) -> LLMResponse:
        self.call_history.append({
            "messages": messages,
            "model": model,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        if self.error:
            return LLMResponse(error=self.error)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, LLMResponse):
                return item
            if isinstance(item, str):
                return LLMResponse(content=item)
        return LLMResponse(content="Default mock response")


# ===========================================================================
# 1. BaseAgent and Architecture Tests
# ===========================================================================

class TestBaseAgentArchitecture:
    def test_base_agent_is_abstract(self):
        """BaseAgent cannot be instantiated directly without implementing run()."""
        with pytest.raises(TypeError):
            BaseAgent(name="Abstract", role="test", description="test")

    def test_agents_inherit_from_base_agent(self):
        planner = PlannerAgent(provider=MockTestProvider())
        executor = ExecutorAgent(provider=MockTestProvider())
        researcher = ResearcherAgent(provider=MockTestProvider())

        assert isinstance(planner, BaseAgent)
        assert isinstance(executor, BaseAgent)
        assert isinstance(researcher, BaseAgent)

    def test_agent_identities_are_distinct(self):
        planner = PlannerAgent(provider=MockTestProvider())
        executor = ExecutorAgent(provider=MockTestProvider())
        researcher = ResearcherAgent(provider=MockTestProvider())

        p_id = planner.get_identity()
        e_id = executor.get_identity()
        r_id = researcher.get_identity()

        assert p_id["name"] == "PlannerAgent"
        assert e_id["name"] == "ExecutorAgent"
        assert r_id["name"] == "ResearcherAgent"

        assert "Planning" in p_id["role"]
        assert "Executing" in e_id["role"]
        assert "Research" in r_id["role"]

    def test_specialized_system_prompts_are_unique(self):
        planner = PlannerAgent(provider=MockTestProvider())
        executor = ExecutorAgent(provider=MockTestProvider())
        researcher = ResearcherAgent(provider=MockTestProvider())

        assert "PLANNING" in planner.system_prompt.upper() or "PLANNER" in planner.system_prompt.upper()
        assert "EXECUTING" in executor.system_prompt.upper() or "EXECUTION" in executor.system_prompt.upper()
        assert "RESEARCH" in researcher.system_prompt.upper()

        # All three must have different instructions
        assert planner.system_prompt != executor.system_prompt
        assert planner.system_prompt != researcher.system_prompt
        assert executor.system_prompt != researcher.system_prompt


# ===========================================================================
# 2. PlannerAgent Tests
# ===========================================================================

class TestPlannerAgent:
    def test_planner_default_model_is_model2(self):
        planner = PlannerAgent(provider=MockTestProvider())
        assert planner.model == MODEL2

    def test_planner_generates_structured_plan(self):
        mock_plan = {
            "tasks": [
                {
                    "id": 1,
                    "description": "Fetch company sales figures",
                    "required_tools": [],
                    "dependencies": [],
                },
                {
                    "id": 2,
                    "description": "Calculate quarterly revenue",
                    "required_tools": ["calculator"],
                    "dependencies": [1],
                },
            ],
            "strategy_notes": "Sequential execution with precision math",
        }
        provider = MockTestProvider(responses=[json.dumps(mock_plan)])
        planner = PlannerAgent(provider=provider)

        result = planner.plan("Analyze Q3 earnings")

        assert "tasks" in result
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["id"] == 1
        assert result["tasks"][0]["description"] == "Fetch company sales figures"
        assert result["tasks"][1]["required_tools"] == ["calculator"]
        assert result["tasks"][1]["dependencies"] == [1]
        assert result["strategy_notes"] == "Sequential execution with precision math"

    def test_planner_does_not_call_tools(self):
        """Planner must plan tasks, not execute tools."""
        mock_plan = {"tasks": [{"id": 1, "description": "Do task"}]}
        provider = MockTestProvider(responses=[json.dumps(mock_plan)])
        planner = PlannerAgent(provider=provider)

        planner.plan("Any objective")

        # Verify LLM was called with tools=None and tool_choice='none'
        assert provider.call_history[0]["tools"] is None
        assert provider.call_history[0]["tool_choice"] == "none"

    def test_planner_handles_empty_objective(self):
        planner = PlannerAgent(provider=MockTestProvider())
        result = planner.plan("")

        assert result["tasks"] == []
        assert "error" in result
        assert "Objective cannot be empty" in result["error"]

    def test_planner_handles_provider_error(self):
        provider = MockTestProvider(error="Network timeout reaching LLM")
        planner = PlannerAgent(provider=provider)

        result = planner.plan("Decompose task")

        assert result["tasks"] == []
        assert "error" in result
        assert "Network timeout" in result["error"]

    def test_planner_run_method_returns_dict(self):
        mock_plan = {"tasks": [{"id": 1, "description": "Step 1"}]}
        provider = MockTestProvider(responses=[json.dumps(mock_plan)])
        planner = PlannerAgent(provider=provider)

        res = planner.run("Plan my day")
        assert isinstance(res, dict)
        assert len(res["tasks"]) == 1


# ===========================================================================
# 3. ExecutorAgent Tests
# ===========================================================================

class TestExecutorAgent:
    def test_executor_default_model_is_model1(self):
        executor = ExecutorAgent(provider=MockTestProvider())
        assert executor.model == MODEL1

    def test_executor_has_authorized_tools(self):
        executor = ExecutorAgent(provider=MockTestProvider())
        assert executor.tools is not None
        tool_names = [t["function"]["name"] for t in executor.tools]
        assert "calculator" in tool_names
        assert "get_datetime" in tool_names
        assert "word_counter" in tool_names

    def test_executor_executes_string_task(self):
        provider = MockTestProvider(responses=["The calculated sum is 42."])
        executor = ExecutorAgent(provider=provider)

        result = executor.execute("Add 40 and 2")

        assert result["status"] == "completed"
        assert result["result"] == "The calculated sum is 42."
        assert result["error"] is None

    def test_executor_executes_structured_dict_task(self):
        provider = MockTestProvider(responses=["Datetime retrieved: 2026-11-15"])
        executor = ExecutorAgent(provider=provider)

        structured_task = {"id": 3, "description": "Get current timestamp"}
        result = executor.execute(structured_task)

        assert result["task_id"] == 3
        assert result["task"] == "Get current timestamp"
        assert result["status"] == "completed"
        assert "2026-11-15" in result["result"]

    def test_executor_handles_empty_task(self):
        executor = ExecutorAgent(provider=MockTestProvider())

        res_str = executor.execute("")
        assert res_str["status"] == "failed"
        assert "cannot be empty" in res_str["error"]

        res_dict = executor.execute({"id": 1, "description": "  "})
        assert res_dict["status"] == "failed"
        assert "cannot be empty" in res_dict["error"]

    def test_executor_handles_provider_error(self):
        provider = MockTestProvider(error="Service unavailable")
        executor = ExecutorAgent(provider=provider)

        result = executor.execute("Execute something")

        assert result["status"] == "failed"
        assert "Service unavailable" in result["result"]
        assert result["error"] is not None

    def test_executor_run_returns_string(self):
        provider = MockTestProvider(responses=["Success result text"])
        executor = ExecutorAgent(provider=provider)

        output = executor.run("Perform operation")
        assert isinstance(output, str)
        assert output == "Success result text"


# ===========================================================================
# 4. ResearcherAgent Tests
# ===========================================================================

class TestResearcherAgent:
    def test_researcher_initialization(self):
        researcher = ResearcherAgent(provider=MockTestProvider())
        assert researcher.name == "ResearcherAgent"
        assert researcher.model == MODEL1
        assert researcher.tools is not None

    def test_researcher_handles_empty_query(self):
        researcher = ResearcherAgent(provider=MockTestProvider())
        result = researcher.research("")

        assert result["status"] == "failed"
        assert "cannot be empty" in result["error"]
        assert result["findings"] == ""

    def test_researcher_executes_research_query(self):
        sample_report = (
            "### Summary of Findings\n"
            "AI agents use tool registries for execution.\n\n"
            "### Verified Findings\n"
            "- AST calculator verifies calculations.\n\n"
            "### Limitations\n"
            "- Web search tool is offline."
        )
        provider = MockTestProvider(responses=[sample_report])
        researcher = ResearcherAgent(provider=provider)

        result = researcher.research("Examine agent tool mechanisms")

        assert result["status"] == "completed"
        assert "Summary of Findings" in result["findings"]
        assert result["error"] is None

    def test_researcher_handles_provider_error(self):
        provider = MockTestProvider(error="Rate limit exceeded")
        researcher = ResearcherAgent(provider=provider)

        result = researcher.research("Research AI security")

        assert result["status"] == "failed"
        assert "Rate limit exceeded" in result["error"]

    def test_researcher_run_returns_findings_string(self):
        provider = MockTestProvider(responses=["Synthesized research output"])
        researcher = ResearcherAgent(provider=provider)

        output = researcher.run("Query topic")
        assert isinstance(output, str)
        assert output == "Synthesized research output"


# ===========================================================================
# 5. Responsibility Boundary and Separation Tests
# ===========================================================================

class TestResponsibilityBoundaries:
    def test_planner_does_not_execute_tools(self):
        """Planner must not invoke tools directly."""
        planner = PlannerAgent(provider=MockTestProvider(responses=['{"tasks": []}']))
        planner.plan("Calculate something")
        assert planner.tools is None

    def test_executor_uses_registered_tools_only(self):
        """Executor must have access only to registered canonical tools."""
        executor = ExecutorAgent(provider=MockTestProvider())
        assert executor.tools is not None
        registered_names = {t["function"]["name"] for t in executor.tools}
        assert registered_names == {"calculator", "get_datetime", "word_counter"}

    def test_backward_compatibility_functional_apis(self):
        """Verify all legacy functions in agents package are importable and functional."""
        from agents import (
            create_plan,
            create_research_plan,
            validate_plan,
            execute_task,
            execute_research_task,
            run_multi_step_agent,
            run_research_agent,
        )

        assert callable(create_plan)
        assert callable(create_research_plan)
        assert callable(validate_plan)
        assert callable(execute_task)
        assert callable(execute_research_task)
        assert callable(run_multi_step_agent)
        assert callable(run_research_agent)
