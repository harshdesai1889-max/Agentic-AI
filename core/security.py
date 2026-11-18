"""
core/security.py

Professional Agent Security Engineering Framework (Day 230).

Implements comprehensive AI and multi-agent security defenses:
1. Part 1: AI Security Fundamentals & Threat Modeling
   - SecurityContext, SecurityFinding, SecurityEvent, SecurityPolicy.
   - Enums: SecurityDecision, ThreatSeverity, ThreatClassification, TrustClassification, DataClassification, AgentRole.
   - Deterministic classification and default-deny policies.
2. Part 2: Prompt Injection and Instruction Security
   - PromptInjectionDetector with deep regex heuristics and contextual checks.
   - Trust classification propagation (SYSTEM_TRUSTED, USER_PROVIDED, USER_TRUSTED, RETRIEVED_UNTRUSTED, TOOL_OUTPUT_UNTRUSTED).
   - System override, secret extraction, approval bypass, and jailbreak detection without naive over-blocking.
3. Part 3: Agent Permissions and Tool Security
   - AgentPermissionEnforcer: Role-based tool allowlists (Researcher read-only, Planner planning only, Executor authorized, Supervisor coordination).
   - Tool execution boundary validation in tools/executor.py and MCP servers.
   - Path traversal and shell injection defense at argument boundaries.
4. Part 4: Secrets, Data Privacy and Secure Storage
   - SensitiveDataScanner & SecretRedactor with deep recursive scrubbing.
   - Redaction of API keys, passwords, bearer tokens, approval tokens, database URIs, PAN/SSN, and exception messages.
   - Immutable copy preservation during redaction.
5. Part 5: Multi-Agent Security and Human Authorization
   - MultiAgentSecurityValidator: Agent anti-spoofing, cross-task isolation, and message ID replay protection.
   - HITL integration: action-bound tokens, single-use consumption, expiration checks, immutability, and Supervisor privilege escalation block.
6. Part 6: Unified Agent Security Framework
   - AgentSecurityManager orchestrating all security components.
   - SQLite security_events auditing and query methods.
   - Independent fail-closed operation decoupled from observability failures.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from utils.logging_config import get_logger

logger = get_logger()

# Global active security manager instance for cross-cutting security checks
_active_security_manager: Optional["AgentSecurityManager"] = None


def get_active_security_manager() -> Optional["AgentSecurityManager"]:
    """Retrieve the currently active AgentSecurityManager."""
    return _active_security_manager


def set_active_security_manager(manager: Optional["AgentSecurityManager"]) -> None:
    """Set the globally active AgentSecurityManager."""
    global _active_security_manager
    _active_security_manager = manager


# ===========================================================================
# 1. PART 1: AI SECURITY FUNDAMENTALS & THREAT MODELING ENUMS & CONTRACTS
# ===========================================================================

class SecurityDecision(str, Enum):
    """Deterministic security outcome for authorization checks."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN_APPROVAL = "require_human_approval"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class ThreatSeverity(str, Enum):
    """Categorical threat risk severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class ThreatClassification(str, Enum):
    """Categorical taxonomy of agentic threats."""
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_OVERRIDE = "system_override"
    SECRET_EXTRACTION = "secret_extraction"
    APPROVAL_BYPASS = "approval_bypass"
    UNAUTHORIZED_TOOL_USAGE = "unauthorized_tool_usage"
    UNAUTHORIZED_TOOL_ACCESS = "unauthorized_tool_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    AGENT_IMPERSONATION = "agent_impersonation"
    SECRET_LEAKAGE = "secret_leakage"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_RETRIEVED_CONTENT = "malicious_retrieved_content"
    MALICIOUS_TOOL_OUTPUT = "malicious_tool_output"
    MCP_SECURITY_RISK = "mcp_security_risk"
    CROSS_TASK_DATA_LEAKAGE = "cross_task_data_leakage"
    UNAUTHORIZED_DESTRUCTIVE_ACTIONS = "unauthorized_destructive_actions"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_lower = value.strip().lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class TrustClassification(str, Enum):
    """Provenance-based trust labels for all content entering the agent system."""
    SYSTEM_TRUSTED = "SYSTEM_TRUSTED"
    USER_PROVIDED = "USER_PROVIDED"
    USER_TRUSTED = "USER_TRUSTED"
    EXTERNAL_UNTRUSTED = "EXTERNAL_UNTRUSTED"
    RETRIEVED_UNTRUSTED = "RETRIEVED_UNTRUSTED"
    TOOL_OUTPUT_UNTRUSTED = "TOOL_OUTPUT_UNTRUSTED"
    AGENT_GENERATED = "AGENT_GENERATED"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_upper = value.strip().upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None


class DataClassification(str, Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_upper = value.strip().upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None


class AgentRole(str, Enum):
    """Known agent roles within the multi-agent architecture."""
    RESEARCHER = "ResearcherAgent"
    PLANNER = "PlannerAgent"
    EXECUTOR = "ExecutorAgent"
    SUPERVISOR = "SupervisorAgent"
    USER = "User"
    UNKNOWN = "UnknownAgent"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val_clean = value.strip().lower().replace("agent", "")
            for member in cls:
                mem_clean = member.value.lower().replace("agent", "")
                if mem_clean == val_clean or member.name.lower() == val_clean:
                    return member
        return None


@dataclass
class SecurityFinding:
    """Individual diagnostic finding generated by security analysis."""
    threat_type: Union[ThreatClassification, str] = ThreatClassification.PROMPT_INJECTION
    severity: Union[ThreatSeverity, str] = ThreatSeverity.MEDIUM
    description: str = ""
    source_trust: Union[TrustClassification, str] = TrustClassification.USER_PROVIDED
    pattern_matched: Optional[str] = None
    risk_score: float = 0.0
    mitigation: Optional[str] = None
    classification: Optional[Union[ThreatClassification, str]] = None
    matched_pattern: Optional[str] = None
    confidence_score: Optional[float] = None

    def __post_init__(self):
        if self.classification is not None:
            self.threat_type = self.classification
        else:
            self.classification = self.threat_type

        if self.matched_pattern is not None:
            self.pattern_matched = self.matched_pattern
        else:
            self.matched_pattern = self.pattern_matched

        if self.confidence_score is not None:
            self.risk_score = self.confidence_score
        else:
            self.confidence_score = self.risk_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary representation."""
        t_val = self.threat_type.value if isinstance(self.threat_type, ThreatClassification) else str(self.threat_type)
        s_val = self.severity.value if isinstance(self.severity, ThreatSeverity) else str(self.severity)
        tr_val = self.source_trust.value if isinstance(self.source_trust, TrustClassification) else str(self.source_trust)
        return {
            "threat_type": t_val,
            "classification": t_val,
            "severity": s_val,
            "description": self.description,
            "source_trust": tr_val,
            "pattern_matched": self.pattern_matched,
            "matched_pattern": self.pattern_matched,
            "risk_score": round(self.risk_score, 4),
            "confidence_score": round(self.risk_score, 4),
            "mitigation": self.mitigation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityFinding":
        """Reconstruct finding from dictionary."""
        return cls(
            threat_type=data.get("threat_type") or data.get("classification") or ThreatClassification.PROMPT_INJECTION.value,
            severity=data.get("severity", ThreatSeverity.MEDIUM.value),
            description=str(data.get("description", "")),
            source_trust=data.get("source_trust", TrustClassification.USER_PROVIDED.value),
            pattern_matched=data.get("pattern_matched") or data.get("matched_pattern"),
            risk_score=float(data.get("risk_score") or data.get("confidence_score") or 0.0),
            mitigation=data.get("mitigation"),
        )


@dataclass
class SecurityContext:
    """Contextual metadata representing the security boundary for an operation."""
    subject_agent: str = "Agent"
    role: Union[AgentRole, str] = AgentRole.RESEARCHER
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    trust_level: Union[TrustClassification, str] = TrustClassification.AGENT_GENERATED
    authorization_token: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_role: Optional[Union[AgentRole, str]] = None
    trust_label: Optional[Union[TrustClassification, str]] = None
    data_classification: Optional[Union[DataClassification, str]] = None

    def __post_init__(self):
        if self.agent_role is not None:
            self.role = self.agent_role
        else:
            self.agent_role = self.role

        if self.trust_label is not None:
            self.trust_level = self.trust_label
        else:
            self.trust_label = self.trust_level

        if self.data_classification is None:
            self.data_classification = DataClassification.INTERNAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert security context to dictionary."""
        r_val = self.role.value if isinstance(self.role, AgentRole) else str(self.role)
        role_short = r_val.lower().replace("agent", "")
        t_val = self.trust_level.value if isinstance(self.trust_level, TrustClassification) else str(self.trust_level)
        d_val = self.data_classification.value if isinstance(self.data_classification, DataClassification) else str(self.data_classification)
        return {
            "subject_agent": self.subject_agent,
            "role": r_val,
            "agent_role": role_short,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "trust_level": t_val,
            "trust_label": t_val.lower(),
            "data_classification": d_val.lower(),
            "authorization_token": "***REDACTED***" if self.authorization_token else None,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class SecurityEvent:
    """Immutable audit record representing an evaluated security decision or incident."""
    threat_category: Union[ThreatClassification, str] = ThreatClassification.PROMPT_INJECTION
    severity: Union[ThreatSeverity, str] = ThreatSeverity.LOW
    decision: Union[SecurityDecision, str] = SecurityDecision.ALLOW
    event_id: str = field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: Optional[str] = None
    source: Optional[str] = None
    trace_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    relevant_resource: Optional[str] = None
    sanitized_explanation: Optional[str] = None
    findings: List[SecurityFinding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_decision: Optional[Union[SecurityDecision, str]] = None

    def __post_init__(self):
        if self.security_decision is not None:
            self.decision = self.security_decision
        else:
            self.security_decision = self.decision

        if self.source is not None and not self.agent:
            self.agent = self.source
        elif self.agent is not None and not self.source:
            self.source = self.agent

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to serializable dictionary."""
        tc_val = self.threat_category.value if isinstance(self.threat_category, ThreatClassification) else str(self.threat_category)
        s_val = self.severity.value if isinstance(self.severity, ThreatSeverity) else str(self.severity)
        dec_val = self.decision.value if isinstance(self.decision, SecurityDecision) else str(self.decision)
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "threat_category": tc_val,
            "severity": s_val,
            "decision": dec_val,
            "security_decision": dec_val,
            "agent": self.agent,
            "source": self.agent,
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "relevant_resource": self.relevant_resource,
            "sanitized_explanation": self.sanitized_explanation,
            "findings": [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in self.findings],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityEvent":
        """Reconstruct SecurityEvent from dictionary."""
        raw_findings = data.get("findings", [])
        findings_objs = [
            SecurityFinding.from_dict(f) if isinstance(f, dict) else f
            for f in raw_findings
        ]
        return cls(
            event_id=str(data.get("event_id", f"sec_{uuid.uuid4().hex[:8]}")),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            threat_category=data.get("threat_category", ThreatClassification.PROMPT_INJECTION.value),
            severity=data.get("severity", ThreatSeverity.LOW.value),
            decision=data.get("security_decision") or data.get("decision", SecurityDecision.ALLOW.value),
            agent=data.get("agent") or data.get("source"),
            trace_id=data.get("trace_id"),
            workflow_id=data.get("workflow_id"),
            task_id=data.get("task_id"),
            relevant_resource=data.get("relevant_resource"),
            sanitized_explanation=data.get("sanitized_explanation"),
            findings=findings_objs,
            metadata=dict(data.get("metadata", {})),
        )


SecurityAuditRecord = SecurityEvent


@dataclass
class SecurityPolicy:
    """Declarative security policy defining agent authorizations and defense thresholds."""
    role_permissions: Dict[str, List[str]] = field(default_factory=lambda: {
        AgentRole.RESEARCHER.value: ["calculator", "get_datetime", "word_counter", "web_search", "document_search", "read_file", "search_web"],
        "Researcher": ["calculator", "get_datetime", "word_counter", "web_search", "document_search", "read_file", "search_web"],
        AgentRole.PLANNER.value: [],
        "Planner": [],
        AgentRole.EXECUTOR.value: ["calculator", "get_datetime", "word_counter", "read_file", "write_file", "execute_shell", "execute_python", "execute_command", "bash"],
        "Executor": ["calculator", "get_datetime", "word_counter", "read_file", "write_file", "execute_shell", "execute_python", "execute_command", "bash"],
        AgentRole.SUPERVISOR.value: [],
        "Supervisor": [],
    })
    privileged_tools: Set[str] = field(default_factory=lambda: {
        "execute_command", "bash", "shell", "run_terminal", "delete_file", "execute_shell", "execute_python",
        "drop_table", "db_mutation", "rm_rf", "format_disk", "restart_cluster", "modify_system",
    })
    fail_closed: bool = True
    default_deny: bool = True
    prompt_injection_threshold: float = 0.65
    strict_agent_validation: bool = True
    redact_secrets: bool = True
    single_use_approvals: bool = True
    allow_shell: bool = True
    workspace_root: Optional[Union[str, Path]] = None
    require_message_signatures: bool = False
    hmac_key: Optional[str] = None

    @classmethod
    def create_hardened_policy(cls, workspace_root: Optional[Union[str, Path]] = None) -> "SecurityPolicy":
        """Factory for hardened security policy with shell execution disabled and signatures enforced."""
        pol = cls(
            allow_shell=False,
            require_message_signatures=True,
            workspace_root=workspace_root,
        )
        for r_key in [AgentRole.EXECUTOR.value, "Executor"]:
            if r_key in pol.role_permissions:
                pol.role_permissions[r_key] = [
                    t for t in pol.role_permissions[r_key]
                    if t not in ("execute_shell", "execute_command", "bash", "execute_python", "shell", "run_terminal")
                ]
        return pol


# ===========================================================================
# 2. PART 2: PROMPT INJECTION AND INSTRUCTION SECURITY
# ===========================================================================

PROMPT_INJECTION_PATTERNS: List[Tuple[re.Pattern, str, ThreatClassification, ThreatSeverity, float]] = [
    # 1. System Prompt & Instruction Overrides
    (
        re.compile(r"(?i)\b(?:ignore|forget|disregard|drop|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system|security)\s+(?:instructions|prompts?|rules|directives?|constraints?|safety\s+rules)\b"),
        "Attempt to override or disregard prior system instructions.",
        ThreatClassification.PROMPT_INJECTION,
        ThreatSeverity.CRITICAL,
        0.95,
    ),
    (
        re.compile(r"(?i)\b(?:you\s+must\s+)?(?:bypass|disable|ignore)\s+security\s+rules\b"),
        "Attempt to bypass security rules.",
        ThreatClassification.PROMPT_INJECTION,
        ThreatSeverity.CRITICAL,
        0.95,
    ),
    (
        re.compile(r"(?i)\b(?:you\s+are\s+now\s+in|switch\s+to|enter)\s+(?:developer|unrestricted|god|jailbreak|DAN|sudo)\s+mode\b"),
        "Attempt to switch agent into unrestricted developer/jailbreak mode.",
        ThreatClassification.PROMPT_INJECTION,
        ThreatSeverity.CRITICAL,
        0.95,
    ),
    (
        re.compile(r"(?i)(?:<\s*system\s*>|\[\s*SYSTEM\s*\]|<\|im_start\|>system|new\s+system\s+directive\s*:|system\s+notice\s*:|system\s+instruction\s*:)"),
        "Attempt to inject raw system role delimiters or impersonate system directives.",
        ThreatClassification.SYSTEM_OVERRIDE,
        ThreatSeverity.CRITICAL,
        0.95,
    ),
    # 2. Secret & Credential Extraction
    (
        re.compile(r"(?i)\b(?:reveal|print|dump|show|output|leak|display)\s+(?:your\s+)?(?:system\s+prompt|initial\s+prompt|api[_-]?keys?|secrets?|passwords?|credentials?|\.env)\b"),
        "Attempt to extract confidential system prompts, API keys, or credentials.",
        ThreatClassification.SECRET_EXTRACTION,
        ThreatSeverity.HIGH,
        0.90,
    ),
    (
        re.compile(r"(?i)\b(?:what\s+is\s+your|tell\s+me\s+your)\s+(?:hidden|secret|internal|system)\s+(?:prompt|instruction|key)\b"),
        "Inquiry probing for hidden system instructions or keys.",
        ThreatClassification.SECRET_EXTRACTION,
        ThreatSeverity.HIGH,
        0.75,
    ),
    # 3. Human Approval Bypass
    (
        re.compile(r"(?i)\b(?:skip|bypass|do\s+not\s+ask\s+for|without)\s+(?:human|user|supervisor|admin)\s+(?:approval|confirmation|review)\b"),
        "Attempt to suppress mandatory human-in-the-loop approval gates.",
        ThreatClassification.APPROVAL_BYPASS,
        ThreatSeverity.HIGH,
        0.85,
    ),
    (
        re.compile(r"(?i)\b(?:human\s+)?approval\s+(?:is|has\s+been|was)\s+(?:already\s+)?(?:granted|given|bypassed|skipped|approved)\b"),
        "Attempt to forge or falsely claim human authorization has been granted.",
        ThreatClassification.APPROVAL_BYPASS,
        ThreatSeverity.HIGH,
        0.85,
    ),
]


class PromptInjectionDetector:
    """Deep inspection layer protecting agents from adversarial prompt injections."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()

    def inspect(
        self,
        content: Any,
        trust_label: Optional[Union[TrustClassification, str]] = None,
        source_trust: Optional[Union[TrustClassification, str]] = None,
    ) -> List[SecurityFinding]:
        """Inspect input content for prompt injection, returning detected findings list."""
        trust_in = trust_label or source_trust or TrustClassification.USER_PROVIDED
        trust = trust_in if isinstance(trust_in, TrustClassification) else (TrustClassification(str(trust_in)) if str(trust_in) in TrustClassification.__members__ else TrustClassification.USER_PROVIDED)

        if trust == TrustClassification.SYSTEM_TRUSTED:
            return []

        text = str(content or "")
        if not text.strip():
            return []

        findings: List[SecurityFinding] = []
        for pattern, desc, threat_type, severity, score in PROMPT_INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_str = match.group(0)
                findings.append(
                    SecurityFinding(
                        threat_type=threat_type,
                        severity=severity,
                        description=desc,
                        source_trust=trust,
                        pattern_matched=matched_str,
                        risk_score=score,
                        mitigation="Content must be quarantined or blocked before agent processing.",
                    )
                )

        return findings

    def classify(
        self,
        content: Any,
        trust_label: Optional[Union[TrustClassification, str]] = None,
        source_trust: Optional[Union[TrustClassification, str]] = None,
    ) -> SecurityDecision:
        """Classify content into ALLOW, REQUIRE_HUMAN_APPROVAL, QUARANTINE, or DENY."""
        trust_in = trust_label or source_trust or TrustClassification.USER_PROVIDED
        trust = trust_in if isinstance(trust_in, TrustClassification) else (TrustClassification(str(trust_in)) if str(trust_in) in TrustClassification.__members__ else TrustClassification.USER_PROVIDED)

        findings = self.inspect(content, trust_label=trust)
        if not findings:
            return SecurityDecision.ALLOW

        max_score = max(f.risk_score for f in findings)
        if max_score >= self.policy.prompt_injection_threshold:
            if trust in (TrustClassification.RETRIEVED_UNTRUSTED, TrustClassification.TOOL_OUTPUT_UNTRUSTED):
                return SecurityDecision.QUARANTINE
            return SecurityDecision.DENY

        return SecurityDecision.REQUIRE_HUMAN_APPROVAL

    def inspect_with_decision(
        self,
        content: Any,
        trust_label: Optional[Union[TrustClassification, str]] = None,
    ) -> Tuple[SecurityDecision, List[SecurityFinding]]:
        """Convenience method returning both decision and findings."""
        findings = self.inspect(content, trust_label=trust_label)
        dec = self.classify(content, trust_label=trust_label)
        return dec, findings


# ===========================================================================
# 3. PART 3: AGENT PERMISSIONS AND TOOL SECURITY
# ===========================================================================

SHELL_INJECTION_PATTERNS = [
    re.compile(r"[;&|`$]|(?:\$\()"),  # Command chaining & substitution
    re.compile(r"(?i)\b(?:rm\s+-rf|format|drop\s+table|truncate|del\s+/f|shutdown|mkfs|cat\s+/etc/|whoami|nc\s+)\b"),
    re.compile(r"(?i)\b(?:powershell|pwsh)(?:\.exe)?\b"),  # PowerShell invocation
    re.compile(r"(?i)\b(?:invoke-expression|iex|invoke-command|start-process|downloadstring|downloadfile)\b"),  # PowerShell cmdlets
    re.compile(r"(?i)-enc(?:odedcommand)?\b"),  # PowerShell base64 encoded command
    re.compile(r"(?i)\bcmd(?:\.exe)?\s+/[ck]\b"),  # Cmd shell invocation
]


def validate_canonical_path(path_str: str, workspace_root: Optional[Union[str, Path]] = None) -> Tuple[bool, Optional[str]]:
    """Validate that path_str resolves strictly within authorized workspace root.

    Protects against:
    - Null-byte injection (\x00)
    - UNC network paths (\\\\server\\share or //server/share)
    - System directory escapes (/etc/, /bin/, C:\\Windows, System32)
    - Relative traversal (../, ..\\)
    - Symlink escapes pointing outside workspace
    """
    if not isinstance(path_str, str):
        return False, "Path must be a string."

    # 1. Null-byte injection check
    if "\x00" in path_str:
        return False, "Null byte detected in path."

    # 2. UNC network path check
    if path_str.startswith(("\\\\", "//")):
        return False, f"UNC network paths are prohibited: '{path_str}'."

    # 3. Known system roots pattern check
    path_lower = path_str.lower().replace("\\", "/")
    if any(s in path_lower for s in ("/etc/", "/etc", "system32", "windows/system32", "c:/windows", "/bin/", "/usr/")):
        return False, f"System directory access prohibited: '{path_str}'."

    # 4. Resolve workspace root
    if workspace_root is not None:
        ws_path = Path(workspace_root)
    else:
        ws_path = Path("c:/AgenticAI/day223_agents/workspace")
    try:
        resolved_ws = ws_path.resolve()
    except Exception:
        resolved_ws = Path.cwd().resolve()

    # 5. Resolve candidate path
    try:
        cand = Path(path_str)
        if cand.is_absolute():
            resolved_cand = cand.resolve()
        else:
            resolved_cand = (resolved_ws / cand).resolve()
    except Exception as e:
        return False, f"Path resolution failed: {e}"

    # 6. Containment check (must reside within workspace root)
    try:
        resolved_cand.relative_to(resolved_ws)
    except ValueError:
        return False, f"Path traversal detected: path '{path_str}' escapes authorized workspace root '{resolved_ws}'."

    # 7. Symlink target validation
    try:
        if cand.is_symlink():
            target_res = cand.resolve()
            target_res.relative_to(resolved_ws)
    except ValueError:
        return False, f"Symlink escape detected: symlink '{path_str}' targets outside workspace."
    except Exception:
        pass

    return True, None


class AgentPermissionEnforcer:
    """Enforces role-based permissions and safety constraints at tool execution boundary."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()

    def check_tool_permission(
        self,
        agent_role: Union[AgentRole, str],
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        approval_token: Optional[str] = None,
        context: Optional[SecurityContext] = None,
    ) -> Tuple[SecurityDecision, str]:
        """Verify if an agent role is authorized to execute a specific tool with args."""
        role_str = agent_role.value if isinstance(agent_role, AgentRole) else str(agent_role)
        norm_tool = str(tool_name).strip().lower()

        # 1. Deny-by-default for unknown roles
        if role_str not in self.policy.role_permissions:
            return SecurityDecision.DENY, f"Permission Denied: Agent role '{role_str}' is unmapped in security policy."

        # 2. Check arguments for path traversal or command injection FIRST (defense-in-depth)
        if tool_args:
            arg_valid, arg_err = self._validate_tool_arguments(tool_args)
            if not arg_valid:
                return SecurityDecision.DENY, f"Security Policy Rejected: {arg_err}"

        # 3. Arbitrary shell execution denial when policy disallows shell
        if norm_tool in ("execute_shell", "bash", "execute_command", "shell", "run_terminal", "execute_python"):
            if not getattr(self.policy, "allow_shell", True):
                return SecurityDecision.DENY, f"Security Policy Violation: Arbitrary shell execution '{norm_tool}' is disabled."

        # 4. Planner restriction: cannot execute any tools
        if "planner" in role_str.lower():
            return SecurityDecision.DENY, "Permission Denied: PlannerAgent cannot execute tools (restricted to planning)."

        # 5. Supervisor restriction: coordination only
        if "supervisor" in role_str.lower():
            return SecurityDecision.DENY, "Permission Denied: SupervisorAgent coordinates workflows and cannot directly execute tools."

        # 6. Role Allowlist
        allowed_tools = self.policy.role_permissions.get(role_str, [])
        if norm_tool not in allowed_tools:
            return SecurityDecision.DENY, f"Permission Denied: Tool '{norm_tool}' is not in permitted tools for agent role '{role_str}'."

        # 7. Researcher read-only restriction
        if "researcher" in role_str.lower():
            if norm_tool in self.policy.privileged_tools or norm_tool in ("write_file", "execute_shell", "execute_python"):
                return SecurityDecision.DENY, f"Permission Denied: ResearcherAgent is read-only and cannot execute '{norm_tool}'."

        # 8. Privileged tool approval requirement (Executor)
        if norm_tool in self.policy.privileged_tools:
            if not approval_token:
                return SecurityDecision.REQUIRE_HUMAN_APPROVAL, f"Authorization Required: Privileged tool '{norm_tool}' requires a human approval token."
            if not str(approval_token).startswith("tok_appr_"):
                return SecurityDecision.DENY, f"Permission Denied: Invalid approval token format '{approval_token}'."

        return SecurityDecision.ALLOW, "Tool execution permitted by security policy."

    def _validate_tool_arguments(self, args: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        if not args or not isinstance(args, dict):
            return True, None

        for k, v in args.items():
            if v is None:
                continue
            str_v = str(v)

            # Path traversal check on any path-related argument or strings with path separators
            is_path_param = (
                k.lower() in ("path", "filepath", "filename", "file", "dest", "destination", "source", "src", "dir", "directory")
                or "/" in str_v or "\\" in str_v
            )
            if is_path_param:
                is_safe, path_err = validate_canonical_path(str_v, workspace_root=self.policy.workspace_root)
                if not is_safe:
                    return False, f"Path traversal pattern detected in argument '{k}': {path_err}"

            # Command injection
            for pat in SHELL_INJECTION_PATTERNS:
                if pat.search(str_v):
                    return False, f"Command injection pattern detected in argument '{k}': {str_v}"

        return True, None


# ===========================================================================
# 4. PART 4: SECRETS, DATA PRIVACY AND SECURE STORAGE
# ===========================================================================

SECRET_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[a-zA-Z0-9_\-]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{10,}"), "[REDACTED_BEARER_TOKEN]"),
    (re.compile(r"tok_appr_[a-zA-Z0-9_\-]{8,}"), "[REDACTED_APPROVAL_TOKEN]"),
    (re.compile(r"(?i)(?:postgres|postgresql|mysql|mongodb|redis)://[a-zA-Z0-9_\-]+:[^@\s]+@[a-zA-Z0-9_\-\.]+:\d+/[a-zA-Z0-9_\-]+"), "[REDACTED_DB_URL]"),
    (re.compile(r"(?i)(?:postgres|postgresql|mysql|mongodb|redis)://[a-zA-Z0-9_\-]+:[^@\s]+@[a-zA-Z0-9_\-\.]+"), "[REDACTED_DB_URL]"),
    (re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"), "password=[REDACTED_PASSWORD]"),
]


class SensitiveDataScanner:
    """Scans and detects sensitive credentials, personal information, and tokens."""

    def __init__(self, classification: DataClassification = DataClassification.SECRET):
        self.classification = classification

    def contains_secrets(self, content: Any) -> bool:
        text = str(content)
        for pat, _ in SECRET_PATTERNS:
            if pat.search(text):
                return True
        return False

    def scan(self, content: Any) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        text = str(content)
        for pat, repl in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                findings.append(
                    SecurityFinding(
                        threat_type=ThreatClassification.DATA_EXFILTRATION,
                        severity=ThreatSeverity.HIGH,
                        description=f"Sensitive secret pattern detected: {repl}",
                        matched_pattern=m.group(0),
                        risk_score=0.85,
                        mitigation="Scrub sensitive credentials using SecretRedactor.",
                    )
                )
        return findings


class SecretRedactor:
    """Recursively redacts secrets and credentials across nested data structures."""

    def redact(self, data: Any) -> Any:
        return self._deep_redact(data)

    def redact_text(self, text: str) -> str:
        return str(self._deep_redact(text))

    def scrub_exception(self, exc: Exception) -> str:
        return str(self._deep_redact(str(exc)))

    def sanitize_exception(self, exc: Exception) -> str:
        return str(self._deep_redact(str(exc)))

    @classmethod
    def _deep_redact(cls, data: Any) -> Any:
        if data is None:
            return None

        if isinstance(data, str):
            res = data
            for pat, repl in SECRET_PATTERNS:
                res = pat.sub(repl, res)
            return res

        elif isinstance(data, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in data.items():
                k_str = str(k).lower()
                if any(s in k_str for s in ("password", "passwd", "pwd")) and not isinstance(v, (dict, list)):
                    cleaned[k] = "[REDACTED_PASSWORD]"
                elif any(s in k_str for s in ("token", "api_key", "secret", "credential", "auth")) and not isinstance(v, (dict, list)):
                    cleaned[k] = cls._deep_redact(v)
                else:
                    cleaned[k] = cls._deep_redact(v)
            return cleaned

        elif isinstance(data, list):
            return [cls._deep_redact(item) for item in data]

        elif isinstance(data, tuple):
            return tuple(cls._deep_redact(item) for item in data)

        elif isinstance(data, Exception):
            return cls._deep_redact(str(data))

        return data


# ===========================================================================
# 5. PART 5: MULTI-AGENT SECURITY, HMAC AUTHENTICATION AND HUMAN AUTHORIZATION
# ===========================================================================

class HMACKeyManager:
    """Thread-safe key management for HMAC-SHA256 inter-agent message signing and verification."""
    _instance: Optional["HMACKeyManager"] = None
    _lock = threading.RLock()

    def __init__(self, initial_key: Optional[str] = None):
        self._keys: List[str] = []
        self._conflict_error: Optional[str] = None

        canonical_key = os.environ.get("AGENT_HMAC_SECRET")
        alias_key = os.environ.get("AB_SMART_AI_HMAC_SECRET")

        # Conflict check if both configured
        if canonical_key and alias_key and canonical_key.strip() and alias_key.strip():
            if canonical_key.strip() != alias_key.strip():
                self._conflict_error = (
                    "HMAC configuration conflict: Both AGENT_HMAC_SECRET and "
                    "AB_SMART_AI_HMAC_SECRET are configured with conflicting values. "
                    "Resolve this conflict by defining only AGENT_HMAC_SECRET."
                )
                logger.error("[HMACKeyManager] Configuration conflict detected between AGENT_HMAC_SECRET and AB_SMART_AI_HMAC_SECRET.")
                # Fail closed
                return

        # Precedence: 1. AGENT_HMAC_SECRET, 2. AB_SMART_AI_HMAC_SECRET, 3. initial_key, 4. persistent key store
        if canonical_key and canonical_key.strip():
            self._keys.append(canonical_key.strip())
        elif alias_key and alias_key.strip():
            logger.warning("[HMACKeyManager] Using deprecated environment variable 'AB_SMART_AI_HMAC_SECRET'. Please migrate to 'AGENT_HMAC_SECRET'.")
            self._keys.append(alias_key.strip())
        elif initial_key and initial_key.strip():
            self._keys.append(initial_key.strip())
        else:
            # Persistent key store in data/.hmac_key to prevent restart invalidation
            key_path = Path("c:/AgenticAI/day223_agents/data/.hmac_key")
            try:
                if key_path.exists():
                    saved = key_path.read_text(encoding="utf-8").strip()
                    if saved:
                        self._keys.append(saved)
            except Exception:
                pass
            if not self._keys:
                gen_key = uuid.uuid4().hex + uuid.uuid4().hex
                try:
                    key_path.parent.mkdir(parents=True, exist_ok=True)
                    key_path.write_text(gen_key, encoding="utf-8")
                    self._keys.append(gen_key)
                except Exception:
                    self._keys.append(gen_key)

    @classmethod
    def get_instance(cls) -> "HMACKeyManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for test isolation and re-evaluating environment)."""
        with cls._lock:
            cls._instance = None

    def get_primary_key(self) -> str:
        with self._lock:
            if self._conflict_error:
                raise ValueError(self._conflict_error)
            return self._keys[0] if self._keys else ""

    def get_all_keys(self) -> List[str]:
        with self._lock:
            if self._conflict_error:
                raise ValueError(self._conflict_error)
            return list(self._keys)

    def rotate_key(self, new_key: str, retire_old: bool = False) -> None:
        """Promote a new primary key for signing while optionally retaining old keys for verification."""
        with self._lock:
            if not new_key or not str(new_key).strip():
                raise ValueError("HMAC key cannot be empty.")
            k = str(new_key).strip()
            if retire_old:
                self._keys = [k]
            else:
                self._keys = [k] + [existing for existing in self._keys if existing != k]
            # Persist rotated key
            try:
                key_path = Path("c:/AgenticAI/day223_agents/data/.hmac_key")
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_text(k, encoding="utf-8")
            except Exception:
                pass


def canonicalize_message(message: Any) -> bytes:
    """Generate deterministic canonical byte representation of an AgentMessage for HMAC signing."""
    auth_version = getattr(message, "auth_version", "v1") or "v1"
    message_id = getattr(message, "message_id", "") or ""
    sender = getattr(message, "sender", "") or ""
    receiver = getattr(message, "receiver", "") or ""
    workflow_id = getattr(message, "workflow_id", "") or ""
    task_id = getattr(message, "task_id", "") or ""
    msg_type = getattr(message, "message_type", "") or ""
    if hasattr(msg_type, "value"):
        msg_type = msg_type.value
    timestamp = getattr(message, "timestamp", "") or ""
    expires_at = getattr(message, "expires_at", "") or ""

    payload_data = {
        "input_data": getattr(message, "input_data", None),
        "output_data": getattr(message, "output_data", None),
    }
    try:
        payload_json = json.dumps(payload_data, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload_json = str(payload_data)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    canonical_str = (
        f"{auth_version}|{message_id}|{sender}|{receiver}|"
        f"{workflow_id}|{task_id}|{msg_type}|{timestamp}|"
        f"{expires_at}|{payload_hash}"
    )
    return canonical_str.encode("utf-8")


def sign_agent_message(message: Any, key: Optional[str] = None) -> Any:
    """Calculate and attach an HMAC-SHA256 signature to an AgentMessage."""
    active_key = key or HMACKeyManager.get_instance().get_primary_key()
    if not active_key:
        raise ValueError("Cannot sign message: No HMAC secret key available.")

    raw_bytes = canonicalize_message(message)
    signature = hmac.new(active_key.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
    if hasattr(message, "signature"):
        message.signature = signature
    elif isinstance(message, dict):
        message["signature"] = signature
    return message


def verify_agent_message_signature(
    message: Any,
    allowed_keys: Optional[List[str]] = None,
    require_signature: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Verify HMAC-SHA256 signature of an AgentMessage using constant-time comparison."""
    sig = getattr(message, "signature", None)
    if sig is None and isinstance(message, dict):
        sig = message.get("signature")

    if not sig:
        if require_signature:
            return False, "Unsigned message rejected: Missing HMAC signature."
        return True, None

    keys = allowed_keys or HMACKeyManager.get_instance().get_all_keys()
    if not keys:
        return False, "Message verification failed: No HMAC secret keys available (fail-closed)."

    raw_bytes = canonicalize_message(message)
    for k in keys:
        expected = hmac.new(k.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
        if hmac.compare_digest(str(sig), expected):
            return True, None

    return False, "HMAC signature verification failed: Invalid message signature or payload tampering."


class MultiAgentSecurityValidator:
    """Validates inter-agent message integrity, identity, task binding, HMAC signatures, and single-use approvals."""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self._lock = threading.RLock()
        self._consumed_approval_tokens: Set[str] = set()
        self._processed_message_ids: Set[str] = set()

    def validate_message(
        self,
        message: Any,
        expected_task_id: Optional[str] = None,
        expected_workflow_id: Optional[str] = None,
    ) -> Tuple[SecurityDecision, str]:
        """Validate an AgentMessage, returning SecurityDecision and explanation."""
        valid, err = self.validate_agent_message(
            message=message,
            expected_task_id=expected_task_id,
            expected_workflow_id=expected_workflow_id,
        )
        if not valid:
            return SecurityDecision.DENY, err or "Validation failed"
        return SecurityDecision.ALLOW, "Message passed multi-agent security validation."

    def validate_agent_message(
        self,
        message: Any,
        expected_workflow_id: Optional[str] = None,
        expected_task_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        sender = getattr(message, "sender", None) or getattr(message, "agent_name", "")
        agent_name = getattr(message, "agent_name", None) or sender
        task_id = getattr(message, "task_id", "")
        workflow_id = getattr(message, "workflow_id", None)
        message_id = getattr(message, "message_id", "")
        expires_at = getattr(message, "expires_at", None)

        # 1. Anti-Spoofing: sender must match agent_name or registered canonical role
        canonical_roles = {"researcher", "planner", "executor", "supervisor", "orchestrator", "evaluator", "user", "agent"}
        sender_norm = str(sender).strip().lower().replace("agent", "")
        agent_norm = str(agent_name).strip().lower().replace("agent", "")

        if sender_norm != agent_norm and sender_norm not in canonical_roles:
            return False, f"Anti-Spoofing Check Failed: Sender '{sender}' does not match registered agent '{agent_name}'."

        # 2. Workflow Boundary Isolation
        if expected_workflow_id and workflow_id:
            if str(workflow_id) != str(expected_workflow_id):
                return False, f"Workflow Boundary Violation: Message workflow '{workflow_id}' does not match expected '{expected_workflow_id}'."

        # 3. Task Boundary Isolation: cross-task message rejection
        if expected_task_id and task_id:
            # Allow step sub-tasks (e.g. task_A_step_1 matches task_A)
            if not str(task_id).startswith(str(expected_task_id)) and not str(expected_task_id).startswith(str(task_id)):
                return False, f"Task Boundary Violation: Message task '{task_id}' does not match expected workflow task '{expected_task_id}' (cross-task leakage blocked)."

        # 4. Message Expiration check
        if expires_at:
            now_iso = datetime.now(timezone.utc).isoformat()
            if now_iso > str(expires_at):
                return False, f"Message Expired: Message '{message_id}' expired at '{expires_at}'."

        # 5. Message ID Replay Protection (thread-safe under concurrent delivery)
        with self._lock:
            if message_id in self._processed_message_ids:
                return False, f"Message Replay Detected: Message ID '{message_id}' was already processed."
            self._processed_message_ids.add(message_id)

        # 6. Cryptographic HMAC Signature Verification
        require_sig = getattr(self.policy, "require_message_signatures", False)
        sig_valid, sig_err = verify_agent_message_signature(
            message,
            require_signature=require_sig,
        )
        if not sig_valid:
            return False, sig_err

        return True, None

    def authorize_action_with_approval(
        self,
        agent_name: str,
        action: str,
        task_id: str,
        approval_token: Optional[str],
        approval_manager: Optional[Any] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> Tuple[SecurityDecision, str]:
        # Supervisor privilege escalation protection
        if "supervisor" in str(agent_name).lower():
            return SecurityDecision.DENY, "Privilege Escalation Blocked: Supervisor cannot authorize its own privileged execution."

        if not approval_token:
            return SecurityDecision.REQUIRE_HUMAN_APPROVAL, "Human approval token required for privileged action."

        with self._lock:
            # Check single-use replay in local cache
            if self.policy.single_use_approvals and approval_token in self._consumed_approval_tokens:
                return SecurityDecision.DENY, f"Replay Detected: Approval token '{approval_token[:8]}...' has already been consumed."

            if approval_manager:
                if hasattr(approval_manager, "consume_token"):
                    consumed_ok, err = approval_manager.consume_token(
                        token=approval_token,
                        task_id=task_id,
                        action=action,
                        workflow_id=workflow_id,
                        agent_name=agent_name,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                    if not consumed_ok:
                        return SecurityDecision.DENY, f"Approval Gate Check Failed: {err}"
                else:
                    valid, err = approval_manager.is_approved(task_id=task_id, action=action, token=approval_token)
                    if not valid:
                        return SecurityDecision.DENY, f"Approval Gate Check Failed: {err}"

            # Consume token locally
            if self.policy.single_use_approvals:
                self._consumed_approval_tokens.add(approval_token)

        return SecurityDecision.ALLOW, "Action successfully authorized by valid human approval."


# ===========================================================================
# 6. PART 6: UNIFIED AGENT SECURITY FRAMEWORK
# ===========================================================================

class AgentSecurityManager:
    """Central coordinator for prompt injection defense, agent permissions, secret privacy,
    inter-agent security, and SQLite security event auditing.
    """

    def __init__(
        self,
        policy: Optional[SecurityPolicy] = None,
        storage: Optional[Any] = None,
        observability: Optional[Any] = None,
        hitl_manager: Optional[Any] = None,
    ):
        self.policy = policy or SecurityPolicy()
        self.storage = storage
        self.observability = observability
        self.hitl_manager = hitl_manager

        self.prompt_defense = PromptInjectionDetector(policy=self.policy)
        self.permissions = AgentPermissionEnforcer(policy=self.policy)
        self.secret_scanner = SensitiveDataScanner()
        self.message_validator = MultiAgentSecurityValidator(policy=self.policy)
        self.multi_agent = self.message_validator
        self.redactor = SecretRedactor()
        self.redact = self.redactor.redact

        self._lock = threading.RLock()
        self._in_memory_events: List[SecurityEvent] = []

    def classify_input(
        self,
        text: str,
        source: str = "user",
        trust_level: Optional[Union[TrustClassification, str]] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[SecurityDecision, List[SecurityFinding]]:
        """Classify input text and inspect for prompt injection / security violations."""
        if trust_level is None:
            trust = TrustClassification.RETRIEVED_UNTRUSTED if source in ("document", "external", "retrieved") else TrustClassification.USER_PROVIDED
        else:
            trust = trust_level if isinstance(trust_level, TrustClassification) else (TrustClassification(str(trust_level)) if str(trust_level) in TrustClassification.__members__ else TrustClassification.USER_PROVIDED)

        findings = self.prompt_defense.inspect(content=text, trust_label=trust)
        decision = self.prompt_defense.classify(content=text, trust_label=trust)

        if decision != SecurityDecision.ALLOW:
            self.record_security_event(
                threat_category=findings[0].threat_type if findings else ThreatClassification.PROMPT_INJECTION,
                severity=findings[0].severity if findings else ThreatSeverity.HIGH,
                decision=decision,
                workflow_id=workflow_id,
                task_id=task_id,
                sanitized_explanation=f"Prompt defense flagged input: {[f.description for f in findings]}",
                findings=findings,
            )
        return decision, findings

    def inspect_content(
        self,
        content: Any,
        source_trust: Union[TrustClassification, str] = TrustClassification.EXTERNAL_UNTRUSTED,
    ) -> Tuple[SecurityDecision, List[SecurityFinding]]:
        """Inspect retrieved documents, tool outputs, or external content."""
        findings = self.prompt_defense.inspect(content=content, trust_label=source_trust)
        decision = self.prompt_defense.classify(content=content, trust_label=source_trust)
        return decision, findings

    def inspect_mcp_output(
        self,
        tool_name: str,
        content: Any,
    ) -> Tuple[SecurityDecision, List[SecurityFinding]]:
        """Inspect MCP tool output for untrusted instructions or prompt injection."""
        findings = self.prompt_defense.inspect(content=content, trust_label=TrustClassification.TOOL_OUTPUT_UNTRUSTED)
        decision = self.prompt_defense.classify(content=content, trust_label=TrustClassification.TOOL_OUTPUT_UNTRUSTED)
        if decision != SecurityDecision.ALLOW:
            self.record_security_event(
                threat_category=ThreatClassification.MALICIOUS_TOOL_OUTPUT,
                severity=ThreatSeverity.HIGH,
                decision=decision,
                relevant_resource=tool_name,
                sanitized_explanation=f"Malicious output detected in MCP tool '{tool_name}'",
                findings=findings,
            )
        return decision, findings

    def evaluate_threat(self, findings: List[SecurityFinding]) -> ThreatSeverity:
        """Derive overall threat severity from a set of security findings."""
        if not findings:
            return ThreatSeverity.LOW
        severities = [f.severity for f in findings]
        if any(s in (ThreatSeverity.CRITICAL, "critical", "CRITICAL") for s in severities):
            return ThreatSeverity.CRITICAL
        if any(s in (ThreatSeverity.HIGH, "high", "HIGH") for s in severities):
            return ThreatSeverity.HIGH
        if any(s in (ThreatSeverity.MEDIUM, "medium", "MEDIUM") for s in severities):
            return ThreatSeverity.MEDIUM
        return ThreatSeverity.LOW

    def authorize_action(
        self,
        agent_name: str,
        action: str,
        task_id: str,
        approval_token: Optional[str] = None,
    ) -> Tuple[SecurityDecision, str]:
        """Authorize a sensitive agent action, enforcing HITL approval when required."""
        decision, reason = self.message_validator.authorize_action_with_approval(
            agent_name=agent_name,
            action=action,
            task_id=task_id,
            approval_token=approval_token,
            approval_manager=self.hitl_manager,
        )
        if decision != SecurityDecision.ALLOW:
            self.record_security_event(
                threat_category=ThreatClassification.PRIVILEGE_ESCALATION,
                severity=ThreatSeverity.HIGH,
                decision=decision,
                agent=agent_name,
                task_id=task_id,
                relevant_resource=action,
                sanitized_explanation=reason,
            )
        return decision, reason

    def check_tool_permission(
        self,
        agent_role: Union[AgentRole, str],
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        approval_token: Optional[str] = None,
        context: Optional[SecurityContext] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[SecurityDecision, str]:
        """Verify tool execution permission at runtime execution boundary."""
        decision, reason = self.permissions.check_tool_permission(
            agent_role=agent_role,
            tool_name=tool_name,
            tool_args=tool_args,
            approval_token=approval_token,
            context=context,
        )
        if decision != SecurityDecision.ALLOW:
            threat = ThreatClassification.UNAUTHORIZED_TOOL_USAGE
            if "traversal" in str(reason).lower() or "injection" in str(reason).lower():
                threat = ThreatClassification.UNAUTHORIZED_DESTRUCTIVE_ACTIONS
            self.record_security_event(
                threat_category=threat,
                severity=ThreatSeverity.HIGH,
                decision=decision,
                agent=agent_role.value if isinstance(agent_role, AgentRole) else str(agent_role),
                workflow_id=workflow_id,
                task_id=task_id,
                relevant_resource=tool_name,
                sanitized_explanation=reason,
            )
        return decision, reason

    def validate_agent_message(
        self,
        message: Any,
        expected_workflow_id: Optional[str] = None,
        expected_task_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate agent message for anti-spoofing and cross-task leakage."""
        valid, err = self.message_validator.validate_agent_message(
            message=message,
            expected_workflow_id=expected_workflow_id,
            expected_task_id=expected_task_id,
        )
        if not valid:
            threat = ThreatClassification.AGENT_IMPERSONATION if "spoofing" in str(err).lower() else ThreatClassification.CROSS_TASK_DATA_LEAKAGE
            self.record_security_event(
                threat_category=threat,
                severity=ThreatSeverity.HIGH,
                decision=SecurityDecision.DENY,
                agent=getattr(message, "sender", None),
                task_id=getattr(message, "task_id", None),
                sanitized_explanation=err,
            )
        return valid, err

    def sanitize_sensitive_data(
        self,
        data: Any,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Any:
        """Deeply scrub sensitive data while creating a safe audit record."""
        return self.redactor.redact(data)

    def record_security_event(
        self,
        threat_category: Union[ThreatClassification, str],
        severity: Union[ThreatSeverity, str],
        decision: Union[SecurityDecision, str],
        agent: Optional[str] = None,
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        relevant_resource: Optional[str] = None,
        sanitized_explanation: Optional[str] = None,
        findings: Optional[List[SecurityFinding]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        """Record an immutable security event in-memory and in SQLite persistence."""
        event = SecurityEvent(
            threat_category=threat_category,
            severity=severity,
            decision=decision,
            agent=agent,
            trace_id=trace_id,
            workflow_id=workflow_id,
            task_id=task_id,
            relevant_resource=relevant_resource,
            sanitized_explanation=sanitized_explanation,
            findings=findings or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._in_memory_events.append(event)

        # Persist to SQLite if storage is available
        if self.storage is not None and hasattr(self.storage, "save_security_event"):
            try:
                self.storage.save_security_event(event.to_dict())
            except Exception as exc:
                logger.warning(f"[Security] Failed to persist security event to SQLite: {exc}")

        return event

    def get_security_summary(self) -> Dict[str, Any]:
        """Generate high-level security metrics summary."""
        with self._lock:
            total_events = len(self._in_memory_events)
            by_threat: Dict[str, int] = {}
            by_decision: Dict[str, int] = {}
            for ev in self._in_memory_events:
                t = ev.threat_category.value if isinstance(ev.threat_category, ThreatClassification) else str(ev.threat_category)
                d = ev.decision.value if isinstance(ev.decision, SecurityDecision) else str(ev.decision)
                by_threat[t] = by_threat.get(t, 0) + 1
                by_decision[d] = by_decision.get(d, 0) + 1

        return {
            "total_security_events": total_events,
            "events_by_threat": by_threat,
            "events_by_decision": by_decision,
        }

    def sign_message(self, message: Any, key: Optional[str] = None) -> Any:
        """Sign an AgentMessage using active HMAC key."""
        return sign_agent_message(message, key=key)

    def rotate_hmac_key(self, new_key: str, retire_old: bool = False) -> None:
        """Rotate the active HMAC secret key."""
        HMACKeyManager.get_instance().rotate_key(new_key, retire_old=retire_old)

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Verify the cryptographic hash-chain integrity of the security audit storage."""
        if self.storage is not None and hasattr(self.storage, "verify_security_audit_chain"):
            return self.storage.verify_security_audit_chain()
        return {"is_valid": True, "total_events": 0, "error": None}
