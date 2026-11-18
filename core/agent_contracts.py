"""
core/agent_contracts.py

Standardized Production Agent Output Contracts for Day 231 (Part 4).

Defines the unified, validated execution contract across ResearcherAgent,
PlannerAgent, ExecutorAgent, and SupervisorAgent.

Guarantees structured outputs with:
- task_id
- workflow_id
- agent_name
- status
- result
- evidence
- errors
- confidence
- execution_metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


@dataclass
class AgentExecutionResult:
    """Standardized output contract for specialized agent task executions."""
    task_id: str
    agent_name: str
    status: str  # "completed" | "failed" | "awaiting_approval" | "in_progress"
    result: Any
    workflow_id: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    errors: Optional[str] = None
    confidence: float = 1.0
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    skills_applied: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "result": self.result,
            "evidence": self.evidence,
            "errors": self.errors,
            "confidence": self.confidence,
            "execution_metadata": self.execution_metadata,
            "skills_applied": list(self.skills_applied),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentExecutionResult:
        """Construct result from dictionary."""
        return cls(
            task_id=str(data.get("task_id", "")),
            agent_name=str(data.get("agent_name", "Agent")),
            status=str(data.get("status", "completed")),
            result=data.get("result"),
            workflow_id=data.get("workflow_id"),
            evidence=data.get("evidence", []) if isinstance(data.get("evidence"), list) else [],
            errors=data.get("errors") or data.get("error"),
            confidence=float(data.get("confidence", 1.0)),
            execution_metadata=data.get("execution_metadata", {}),
            skills_applied=list(data.get("skills_applied", [])),
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        )
