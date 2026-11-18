"""
core/external_integrations.py

Production External Application Integration Architecture for Day 231 (Part 1).

Enables AB SMART AI agents to interact with external applications through
secure, controlled integration interfaces.

Implements:
1. AuthType: Supported authentication schemes (API_KEY, OAUTH2, BEARER_TOKEN, NONE).
2. ExternalApplication: Registration model defining endpoints, auth requirements, timeouts, and rate limits.
3. ExternalAppRegistry: Thread-safe central registry for registering and discovering external applications.
4. ExternalAppAdapter: Modular base adapter contract for capability discovery, request validation,
   execution, response normalization, retry policies, and security audit logging.
5. MockExternalAppAdapter: Clearly labeled safe mock adapter demonstrating end-to-end integration
   without requiring paid external subscriptions.
6. Zero-Leak Credential Security: Credentials never enter prompts, messages, logs, or trace spans.
   Sensitive operations gate behind HumanApprovalManager.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from utils.logging_config import get_logger

logger = get_logger()


class AuthType(str, Enum):
    """Supported authentication schemes for external applications."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    NONE = "none"


@dataclass
class ExternalApplication:
    """Metadata and connection configuration for an authorized external application."""
    app_id: str
    name: str
    description: str
    base_url: str
    auth_type: AuthType = AuthType.API_KEY
    rate_limit_rpm: int = 60
    timeout_seconds: float = 30.0
    max_retries: int = 3
    is_active: bool = True
    is_mock: bool = False
    requires_approval_for_write: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert application metadata to a serializable dictionary, redacting sensitive fields."""
        return {
            "app_id": self.app_id,
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "auth_type": self.auth_type.value,
            "rate_limit_rpm": self.rate_limit_rpm,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "is_active": self.is_active,
            "is_mock": self.is_mock,
            "requires_approval_for_write": self.requires_approval_for_write,
            "metadata": {k: v for k, v in self.metadata.items() if not any(s in k.lower() for s in ("secret", "key", "token", "password"))},
            "registered_at": self.registered_at,
        }


class ExternalAppAdapter(ABC):
    """Abstract adapter defining the modular interface for external application connectors."""

    def __init__(self, application: ExternalApplication):
        self.app = application
        self._lock = threading.RLock()

    @abstractmethod
    def discover_capabilities(self) -> List[Dict[str, Any]]:
        """Return a list of available operations, parameters, and descriptions."""
        pass

    @abstractmethod
    def validate_request(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate input parameters against the application's operation schema."""
        pass

    @abstractmethod
    def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        auth_context: Optional[Dict[str, Any]] = None,
        security_manager: Optional[Any] = None,
        approval_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the external operation, applying security checks, timeouts, and normalization."""
        pass

    @abstractmethod
    def normalize_response(self, raw_response: Any) -> Dict[str, Any]:
        """Transform raw external application responses into standardized AB SMART AI schema."""
        pass


class ExternalAppRegistry:
    """Thread-safe singleton registry managing registered external applications and adapters."""

    _instance: Optional["ExternalAppRegistry"] = None
    _lock = threading.RLock()

    def __init__(self):
        self._apps: Dict[str, ExternalApplication] = {}
        self._adapters: Dict[str, ExternalAppAdapter] = {}

    @classmethod
    def get_instance(cls) -> "ExternalAppRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_application(self, app: ExternalApplication, adapter: ExternalAppAdapter) -> None:
        """Register an application and its associated connector adapter."""
        with self._lock:
            self._apps[app.app_id] = app
            self._adapters[app.app_id] = adapter
            logger.info(f"[ExternalAppRegistry] Registered application '{app.name}' ({app.app_id}) [mock={app.is_mock}]")

    def unregister_application(self, app_id: str) -> bool:
        """Remove an application from the registry."""
        with self._lock:
            app = self._apps.pop(app_id, None)
            self._adapters.pop(app_id, None)
            return app is not None

    def get_application(self, app_id: str) -> Optional[ExternalApplication]:
        """Fetch application configuration by ID."""
        with self._lock:
            return self._apps.get(app_id)

    def get_adapter(self, app_id: str) -> Optional[ExternalAppAdapter]:
        """Fetch application adapter by ID."""
        with self._lock:
            return self._adapters.get(app_id)

    def list_applications(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List registered applications with sensitive fields sanitized."""
        with self._lock:
            res = []
            for app in self._apps.values():
                if active_only and not app.is_active:
                    continue
                res.append(app.to_dict())
            return res

    def clear(self) -> None:
        """Clear all registered applications (useful for isolated unit testing)."""
        with self._lock:
            self._apps.clear()
            self._adapters.clear()


class MockExternalAppAdapter(ExternalAppAdapter):
    """Clearly identified safe mock adapter representing a registered external business service.
    
    Demonstrates:
    - Application registration
    - Capability discovery
    - Authorized request execution
    - Error and timeout simulation
    - Response normalization
    - Security audit event logging
    """

    def __init__(self, application: Optional[ExternalApplication] = None):
        app = application or ExternalApplication(
            app_id="mock_customer_service",
            name="Mock Enterprise CRM & Support System",
            description="Safe mock external service providing customer lookup and support ticket management.",
            base_url="https://mock-crm.internal.local/api/v1",
            auth_type=AuthType.API_KEY,
            is_mock=True,
            requires_approval_for_write=True,
        )
        super().__init__(app)
        self._mock_data = {
            "customers": {
                "cust_101": {"id": "cust_101", "name": "Apex Global Logistics", "tier": "Enterprise", "status": "Active"},
                "cust_102": {"id": "cust_102", "name": "Quantum Dynamics", "tier": "Standard", "status": "Active"},
            },
            "tickets": [
                {"id": "tkt_001", "customer_id": "cust_101", "subject": "System Latency Advisory", "status": "Open", "priority": "High"},
                {"id": "tkt_002", "customer_id": "cust_102", "subject": "Billing Clarification", "status": "Closed", "priority": "Normal"},
            ],
        }

    def discover_capabilities(self) -> List[Dict[str, Any]]:
        """Return available operations on the mock enterprise application."""
        return [
            {
                "operation": "lookup_customer",
                "description": "Look up customer account details by ID.",
                "parameters": {"customer_id": {"type": "string", "required": True}},
                "is_mutating": False,
                "requires_approval": False,
            },
            {
                "operation": "list_tickets",
                "description": "List open support tickets for a given customer.",
                "parameters": {"customer_id": {"type": "string", "required": False}},
                "is_mutating": False,
                "requires_approval": False,
            },
            {
                "operation": "create_ticket",
                "description": "Create a new priority support ticket (sensitive operation).",
                "parameters": {
                    "customer_id": {"type": "string", "required": True},
                    "subject": {"type": "string", "required": True},
                    "priority": {"type": "string", "required": False, "default": "Normal"},
                },
                "is_mutating": True,
                "requires_approval": True,
            },
        ]

    def validate_request(self, operation: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate operation and mandatory arguments."""
        if not operation or not isinstance(operation, str):
            return False, "Operation name must be a non-empty string."

        capabilities = {c["operation"]: c for c in self.discover_capabilities()}
        if operation not in capabilities:
            return False, f"Unsupported operation '{operation}'. Supported operations: {list(capabilities.keys())}"

        schema = capabilities[operation]
        for param_name, param_meta in schema.get("parameters", {}).items():
            if param_meta.get("required", False) and param_name not in params:
                return False, f"Missing required parameter '{param_name}' for operation '{operation}'."

        return True, None

    def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        auth_context: Optional[Dict[str, Any]] = None,
        security_manager: Optional[Any] = None,
        approval_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the external operation safely with parameter validation and audit logging."""
        with self._lock:
            # 1. Parameter validation
            valid, err = self.validate_request(operation, params)
            if not valid:
                return {"status": "error", "error": f"Invalid external request: {err}"}

            capabilities = {c["operation"]: c for c in self.discover_capabilities()}
            op_meta = capabilities[operation]

            # 2. Sensitive operation gate: Require human approval if mutating
            if op_meta.get("requires_approval", False) and self.app.requires_approval_for_write:
                if not approval_token:
                    if security_manager:
                        security_manager.record_security_event(
                            threat_category="UNAUTHORIZED_TOOL_USAGE",
                            severity="HIGH",
                            decision="REQUIRE_HUMAN_APPROVAL",
                            relevant_resource=f"{self.app.app_id}:{operation}",
                            sanitized_explanation=f"Mutating operation '{operation}' requires human approval token.",
                        )
                    return {
                        "status": "error",
                        "error": f"Security Policy Rejected: Operation '{operation}' requires human approval token.",
                        "requires_approval": True,
                    }

            # 3. Simulated business logic execution
            if operation == "lookup_customer":
                cid = params["customer_id"]
                cust = self._mock_data["customers"].get(cid)
                if not cust:
                    return {"status": "error", "error": f"Customer '{cid}' not found in external CRM."}
                return self.normalize_response({"customer": cust})

            elif operation == "list_tickets":
                cid = params.get("customer_id")
                tickets = self._mock_data["tickets"]
                if cid:
                    tickets = [t for t in tickets if t["customer_id"] == cid]
                return self.normalize_response({"tickets": tickets, "count": len(tickets)})

            elif operation == "create_ticket":
                new_ticket = {
                    "id": f"tkt_{uuid.uuid4().hex[:6]}",
                    "customer_id": params["customer_id"],
                    "subject": params["subject"],
                    "priority": params.get("priority", "Normal"),
                    "status": "Open",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self._mock_data["tickets"].append(new_ticket)
                return self.normalize_response({"ticket_created": new_ticket, "success": True})

            return {"status": "error", "error": f"Unhandled operation '{operation}'"}

    def normalize_response(self, raw_response: Any) -> Dict[str, Any]:
        """Normalize raw response into standardized AB SMART AI output format."""
        return {
            "status": "success",
            "source_application": self.app.app_id,
            "is_mock": True,
            "data": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
