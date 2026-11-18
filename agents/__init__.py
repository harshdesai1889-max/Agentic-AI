"""
agents/__init__.py

Specialized AI Agents package for day223_agents.
Exposes specialized agent classes, multi-agent orchestrator,
task manager, and backward-compatible functional wrappers.
"""

from agents.base_agent import BaseAgent
from agents.planner import (
    PlannerAgent,
    create_plan,
    create_research_plan,
    validate_plan,
)
from agents.executor import (
    ExecutorAgent,
    execute_task,
    execute_research_task,
)
from agents.researcher import (
    ResearcherAgent,
    run_multi_step_agent,
    run_research_agent,
)
from agents.supervisor import (
    SupervisorAgent,
    SupervisorDecision,
    DecisionType,
    validate_supervisor_decision,
)
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    AgentMessage,
    WorkflowStatus,
)
from core.task_manager import (
    TaskManager,
    TaskSpecification,
    TaskStatus,
    VerificationResult,
)

__all__ = [
    # Base class
    "BaseAgent",
    # Specialized agent classes
    "PlannerAgent",
    "ExecutorAgent",
    "ResearcherAgent",
    "SupervisorAgent",
    "SupervisorDecision",
    "DecisionType",
    "validate_supervisor_decision",
    # Multi-Agent Workflow Orchestrator & Contracts
    "WorkflowOrchestrator",
    "TaskContext",
    "AgentMessage",
    "WorkflowStatus",
    # Task Management & Assignment
    "TaskManager",
    "TaskSpecification",
    "TaskStatus",
    "VerificationResult",
    # Backward-compatible functional APIs
    "create_plan",
    "create_research_plan",
    "validate_plan",
    "execute_task",
    "execute_research_task",
    "run_multi_step_agent",
    "run_research_agent",
]
