import json
from typing import Any, Dict, Optional, Union
from tools.registry import TOOL_REGISTRY, get_tool


def execute_tool(
    tool_call: Any,
    observability: Optional[Any] = None,
    security: Optional[Any] = None,
    agent_role: Optional[str] = None,
    approval_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single tool call requested by the model.

    Args:
        tool_call: Either an OpenAI tool_call object (with .function.name, .function.arguments)
                   or a dictionary with corresponding keys.
        observability: Optional AgentObservabilityManager to record tool execution span.
        security: Optional AgentSecurityManager to enforce runtime permission checks.
        agent_role: Role of calling agent (e.g. ResearcherAgent, ExecutorAgent).
        approval_token: Cryptographic approval token for privileged operations.

    Returns:
        A JSON-serializable dictionary representing the tool output or an error.
    """
    # 1. Identify the requested tool name and raw arguments
    try:
        if hasattr(tool_call, "function"):
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
        elif hasattr(tool_call, "name") and hasattr(tool_call, "arguments"):
            tool_name = tool_call.name
            raw_arguments = tool_call.arguments
        elif isinstance(tool_call, dict):
            if "function" in tool_call and isinstance(tool_call["function"], dict):
                tool_name = tool_call["function"].get("name")
                raw_arguments = tool_call["function"].get("arguments", "{}")
            else:
                tool_name = tool_call.get("name") or tool_call.get("tool_name")
                raw_arguments = tool_call.get("arguments") if tool_call.get("arguments") is not None else (tool_call.get("parameters") or "{}")
        else:
            return {"error": "Invalid tool call format"}
    except Exception:
        return {"error": "Failed to parse tool call"}

    # 2. Parse JSON arguments
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}
    else:
        return {"error": "Invalid tool arguments"}

    # 3. Validate arguments is a dictionary
    if not isinstance(arguments, dict):
        return {"error": "Invalid tool arguments"}

    norm_name = str(tool_name or "").strip().lower()

    # 4. Strict Shell Execution Sandboxing: Arbitrary shell execution is disabled by default
    if norm_name in ("execute_shell", "bash", "execute_command", "shell", "run_terminal", "execute_python"):
        return {
            "error": "Security policy rejected tool execution: Arbitrary shell execution is disabled by policy.",
            "status": "error",
        }

    # 5. Enforce security permission if explicitly provided
    if security is not None:
        role = agent_role or "ExecutorAgent"
        decision, perm_err = security.check_tool_permission(
            agent_role=role,
            tool_name=tool_name or "",
            tool_args=arguments,
            approval_token=approval_token,
        )
        is_allowed = (decision == "allow" or getattr(decision, "value", None) == "allow" or decision is True)
        if not is_allowed:
            return {"error": f"Security policy rejected tool execution: {perm_err}", "status": "error"}

    # 6. Check tool availability in registry
    tool_func = get_tool(tool_name)
    if not tool_func:
        return {"error": "Unknown tool requested"}

    # 7. Check ambient security manager if not explicitly provided
    if security is None:
        try:
            from core.security import get_active_security_manager
            amb_sec = get_active_security_manager()
            if amb_sec is not None:
                role = agent_role or "ExecutorAgent"
                decision, perm_err = amb_sec.check_tool_permission(
                    agent_role=role,
                    tool_name=tool_name,
                    tool_args=arguments,
                    approval_token=approval_token,
                )
                is_allowed = (decision == "allow" or getattr(decision, "value", None) == "allow" or decision is True)
                if not is_allowed:
                    return {"error": f"Security policy rejected tool execution: {perm_err}", "status": "error"}
        except Exception:
            pass

    # 8. Resolve observability manager (explicit or ambient)
    obs = observability
    if obs is None:
        try:
            from core.observability import get_active_observability_manager
            obs = get_active_observability_manager()
        except Exception:
            obs = None

    # 7. Execute the registered function safely (instrumented once when obs is present)
    try:
        if obs is not None:
            from core.observability import instrument_tool_execution
            result = instrument_tool_execution(
                tool_func=tool_func,
                tool_name=tool_name,
                tool_args=arguments,
                observability=obs,
            )
        else:
            result = tool_func(**arguments)

        if not isinstance(result, dict):
            result = {"result": result}
        return result
    except TypeError:
        return {"error": "Incorrect tool parameters"}
    except Exception:
        return {"error": "Tool execution failed"}

