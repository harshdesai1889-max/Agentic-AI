"""
core/mcp_connector.py

Production MCP Tool Connectivity & Discovery Architecture for Day 231 (Part 2).

Provides reusable, secure connectivity to Model Context Protocol (MCP) servers,
enforcing discovery, schema validation, connection lifecycle management,
and runtime security boundary enforcement.

Implements:
1. MCPConnectionState: Lifecycle states (DISCONNECTED, CONNECTED, HEALTHY, FAILED).
2. MCPToolMetadata: Validated tool definitions extracted from MCP server instances.
3. MCPToolConnector: High-level client managing connection lifecycle, tool discovery,
   input schema validation, health checks, and secure execution through AgentSecurityManager.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from utils.logging_config import get_logger

logger = get_logger()


class MCPConnectionState(str, Enum):
    """Connection lifecycle states for an MCP server connector."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class MCPToolMetadata:
    """Structured metadata describing an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = "basic-tools-server"
    version: str = "1.0.0"
    is_safe: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
            "version": self.version,
            "is_safe": self.is_safe,
        }


class MCPToolConnector:
    """Manages connection, discovery, schema validation, and secure execution for MCP servers."""

    def __init__(
        self,
        server_instance: Optional[Any] = None,
        server_name: str = "basic-tools-server",
        timeout_seconds: float = 10.0,
    ):
        self.server_name = server_name
        self.timeout_seconds = timeout_seconds
        self._server = server_instance
        self._state = MCPConnectionState.DISCONNECTED
        self._discovered_tools: Dict[str, MCPToolMetadata] = {}
        self._lock = threading.RLock()
        self._last_health_check: Optional[str] = None
        self._failure_count: int = 0

    @property
    def state(self) -> MCPConnectionState:
        return self._state

    def connect(self) -> bool:
        """Initialize connection to the MCP server and discover registered tools."""
        with self._lock:
            self._state = MCPConnectionState.CONNECTING
            try:
                # Lazy import basic tools server if no explicit instance provided
                if self._server is None:
                    from mcp_servers.basic_tools_server import server
                    self._server = server

                # Perform tool discovery
                tools = self.discover_tools()
                if not tools:
                    self._state = MCPConnectionState.DEGRADED
                    logger.warning(f"[MCPToolConnector] Connected to '{self.server_name}', but 0 tools discovered.")
                else:
                    self._state = MCPConnectionState.HEALTHY
                    logger.info(f"[MCPToolConnector] Successfully connected to '{self.server_name}' ({len(tools)} tools discovered).")
                self._last_health_check = datetime.now(timezone.utc).isoformat()
                self._failure_count = 0
                return True
            except Exception as exc:
                self._state = MCPConnectionState.FAILED
                self._failure_count += 1
                logger.error(f"[MCPToolConnector] Connection failed for '{self.server_name}': {exc}")
                return False

    def disconnect(self) -> None:
        """Gracefully disconnect from the MCP server."""
        with self._lock:
            self._state = MCPConnectionState.DISCONNECTED
            logger.info(f"[MCPToolConnector] Disconnected from '{self.server_name}'.")

    def reconnect(self) -> bool:
        """Attempt reconnection after degradation or failure."""
        self.disconnect()
        return self.connect()

    def health_check(self) -> Dict[str, Any]:
        """Verify the health and responsiveness of the MCP server connection."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            if self._server is None or self._state in (MCPConnectionState.DISCONNECTED, MCPConnectionState.FAILED):
                return {
                    "server_name": self.server_name,
                    "status": "unhealthy",
                    "state": self._state.value,
                    "tool_count": 0,
                    "timestamp": now_iso,
                    "error": "Server not connected",
                }

            try:
                # Verify tool manager is accessible
                tool_list = self.discover_tools()
                self._state = MCPConnectionState.HEALTHY
                self._last_health_check = now_iso
                return {
                    "server_name": self.server_name,
                    "status": "healthy",
                    "state": self._state.value,
                    "tool_count": len(tool_list),
                    "timestamp": now_iso,
                    "error": None,
                }
            except Exception as exc:
                self._state = MCPConnectionState.FAILED
                self._failure_count += 1
                return {
                    "server_name": self.server_name,
                    "status": "unhealthy",
                    "state": self._state.value,
                    "tool_count": 0,
                    "timestamp": now_iso,
                    "error": str(exc),
                }

    def discover_tools(self) -> List[MCPToolMetadata]:
        """Discover tools registered with the MCP server and build validated metadata records."""
        with self._lock:
            if self._server is None:
                return []

            discovered = []
            try:
                # Access MCPServer internal tool manager
                if hasattr(self._server, "_tool_manager") and hasattr(self._server._tool_manager, "list_tools"):
                    raw_tools = self._server._tool_manager.list_tools()
                    for t in raw_tools:
                        name = getattr(t, "name", "")
                        desc = getattr(t, "description", "")
                        schema = getattr(t, "inputSchema", None)
                        if not schema or not isinstance(schema, dict):
                            if name == "calculate":
                                schema = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
                            elif name == "count_words":
                                schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
                            else:
                                schema = {"type": "object", "properties": {}}

                        meta = MCPToolMetadata(
                            name=name,
                            description=desc,
                            input_schema=schema,
                            server_name=self.server_name,
                        )
                        self._discovered_tools[name] = meta
                        discovered.append(meta)
                else:
                    # Fallback to known server tools if reflection not available
                    from tools.basic_tools import calculator, get_datetime, word_counter
                    fallback_tools = [
                        MCPToolMetadata(name="calculate", description="Evaluate a mathematical expression.", input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
                        MCPToolMetadata(name="current_datetime", description="Return the current date and time in ISO format.", input_schema={"type": "object", "properties": {}}),
                        MCPToolMetadata(name="count_words", description="Count words and characters in text.", input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
                    ]
                    for meta in fallback_tools:
                        self._discovered_tools[meta.name] = meta
                        discovered.append(meta)
            except Exception as exc:
                logger.error(f"[MCPToolConnector] Error during tool discovery: {exc}")

            return discovered

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool name and input argument schema before execution."""
        with self._lock:
            if not self._discovered_tools:
                self.discover_tools()

            tool_meta = self._discovered_tools.get(tool_name)
            if not tool_meta:
                return False, f"Unknown MCP tool '{tool_name}'. Available tools: {list(self._discovered_tools.keys())}"

            # Validate required properties from schema
            schema = tool_meta.input_schema
            required_fields = schema.get("required", [])
            for req in required_fields:
                if req not in arguments:
                    return False, f"Missing required argument '{req}' for tool '{tool_name}'."

            return True, None

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_role: str = "ExecutorAgent",
        approval_token: Optional[str] = None,
        security_manager: Optional[Any] = None,
        observability: Optional[Any] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an MCP tool through the full security and observability perimeter."""
        # 1. State check
        if self._state == MCPConnectionState.DISCONNECTED:
            self.connect()

        # 2. Input schema validation
        val_ok, val_err = self.validate_tool_call(tool_name, arguments)
        if not val_ok:
            return {"status": "error", "error": f"Schema validation failed: {val_err}"}

        # 3. Security permission check (AgentSecurityManager / AgentPermissionEnforcer)
        sec = security_manager
        if sec is None:
            try:
                from core.security import get_active_security_manager
                sec = get_active_security_manager()
            except Exception:
                sec = None

        if sec is not None:
            # Map calculate -> calculator for role allowlist consistency
            canonical_tool = "calculator" if tool_name == "calculate" else ("get_datetime" if tool_name == "current_datetime" else ("word_counter" if tool_name == "count_words" else tool_name))
            decision, reason = sec.check_tool_permission(
                agent_role=agent_role,
                tool_name=canonical_tool,
                tool_args=arguments,
                approval_token=approval_token,
                workflow_id=workflow_id,
                task_id=task_id,
            )
            is_allowed = (decision == "allow" or getattr(decision, "value", None) == "allow" or decision is True)
            if not is_allowed:
                return {"status": "error", "error": f"Security policy rejected MCP tool execution: {reason}"}

        # 4. Observability instrumentation
        obs = observability
        if obs is None:
            try:
                from core.observability import get_active_observability_manager
                obs = get_active_observability_manager()
            except Exception:
                obs = None

        # 5. Execution through MCP server or fallback wrapper
        try:
            if self._server is not None and hasattr(self._server, "call_tool"):
                try:
                    result = asyncio.run(self._server.call_tool(tool_name, arguments))
                    parsed_res = self._parse_call_tool_result(result)
                except RuntimeError:
                    # If already in an event loop or async context
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        from mcp_servers.basic_tools_server import calculate, current_datetime, count_words
                        parsed_res = self._direct_call_fallback(tool_name, arguments)
                    else:
                        result = loop.run_until_complete(self._server.call_tool(tool_name, arguments))
                        parsed_res = self._parse_call_tool_result(result)
            else:
                parsed_res = self._direct_call_fallback(tool_name, arguments)

            # 6. Redaction of sensitive values in tool output
            if sec is not None and hasattr(sec, "sanitize_sensitive_data"):
                parsed_res = sec.sanitize_sensitive_data(parsed_res)

            return {
                "status": "success",
                "tool_name": tool_name,
                "server_name": self.server_name,
                "result": parsed_res,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.error(f"[MCPToolConnector] Tool execution failure for '{tool_name}': {exc}")
            return {"status": "error", "error": f"MCP tool execution failed: {str(exc)}"}

    def _parse_call_tool_result(self, result: Any) -> Any:
        """Extract Python result object from an MCP CallToolResult envelope."""
        if hasattr(result, "content") and result.content:
            item = result.content[0]
            text = item.text if hasattr(item, "text") else str(item)
            try:
                return json.loads(text)
            except Exception:
                return text
        return result

    def _direct_call_fallback(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Direct execution fallback mapping to basic_tools implementations."""
        from tools.basic_tools import calculator, get_datetime, word_counter
        if tool_name == "calculate":
            return calculator(arguments.get("expression", ""))
        elif tool_name == "current_datetime":
            return get_datetime()
        elif tool_name == "count_words":
            return word_counter(arguments.get("text", ""))
        else:
            raise ValueError(f"Unknown MCP tool fallback: {tool_name}")
