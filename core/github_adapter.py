"""
core/github_adapter.py

Production Read-Only GitHub REST API Adapter for Day 231.1 (Part 2).

Implements a genuine read-only GitHub integration using ExternalAppAdapter.

Supported operations (GET only):
- get_authenticated_user: Retrieve authenticated user information.
- get_repository: Retrieve repository information.
- list_issues: List repository issues.
- get_issue: Retrieve a specific issue.
- list_pull_requests: List repository pull requests.

Security guarantees:
- Token read from GITHUB_TOKEN environment variable only.
- Token never hardcoded, printed, logged, stored in SQLite, or included in messages.
- Write operations (POST/PUT/PATCH/DELETE) are explicitly rejected at execution boundary.
- All errors sanitized to prevent token leakage.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.external_integrations import (
    AuthType,
    ExternalAppAdapter,
    ExternalApplication,
)
from utils.logging_config import get_logger

logger = get_logger()

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

# Operations that are explicitly allowed (read-only GET operations)
READ_ONLY_OPERATIONS = frozenset({
    "get_authenticated_user",
    "get_repository",
    "list_issues",
    "get_issue",
    "list_pull_requests",
})

# Operations that are explicitly rejected (write operations)
WRITE_OPERATIONS = frozenset({
    "create_issue",
    "edit_issue",
    "delete_issue",
    "create_pull_request",
    "merge_pull_request",
    "modify_repository",
    "write_comment",
    "create_comment",
    "update_comment",
    "delete_comment",
})


class GitHubReadOnlyAdapter(ExternalAppAdapter):
    """Read-only GitHub REST API adapter.

    This adapter enforces read-only access at the execution boundary.
    Write operations are rejected regardless of parameters or tokens.

    Token management:
    - Read from GITHUB_TOKEN environment variable
    - Never stored in SQLite, messages, or logs
    - Redacted from all error messages and responses
    """

    def __init__(
        self,
        application: Optional[ExternalApplication] = None,
        timeout_seconds: float = 15.0,
    ):
        app = application or ExternalApplication(
            app_id="github_readonly",
            name="GitHub REST API (Read-Only)",
            description="Read-only access to GitHub repositories, issues, and pull requests.",
            base_url=GITHUB_API_BASE,
            auth_type=AuthType.BEARER_TOKEN,
            is_mock=False,
            requires_approval_for_write=True,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(app)
        self._timeout = timeout_seconds

    def _get_token(self) -> Optional[str]:
        """Retrieve GitHub token from environment. Never cache or persist."""
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        return token if token else None

    def is_configured(self) -> bool:
        """Check if GitHub token is available."""
        return self._get_token() is not None

    def discover_capabilities(self) -> List[Dict[str, Any]]:
        """Return available read-only operations."""
        return [
            {
                "operation": "get_authenticated_user",
                "description": "Retrieve authenticated user information.",
                "parameters": {},
                "is_mutating": False,
                "requires_approval": False,
                "http_method": "GET",
            },
            {
                "operation": "get_repository",
                "description": "Retrieve repository information.",
                "parameters": {
                    "owner": {"type": "string", "required": True},
                    "repo": {"type": "string", "required": True},
                },
                "is_mutating": False,
                "requires_approval": False,
                "http_method": "GET",
            },
            {
                "operation": "list_issues",
                "description": "List repository issues.",
                "parameters": {
                    "owner": {"type": "string", "required": True},
                    "repo": {"type": "string", "required": True},
                    "state": {"type": "string", "required": False, "default": "open"},
                },
                "is_mutating": False,
                "requires_approval": False,
                "http_method": "GET",
            },
            {
                "operation": "get_issue",
                "description": "Retrieve a specific issue.",
                "parameters": {
                    "owner": {"type": "string", "required": True},
                    "repo": {"type": "string", "required": True},
                    "issue_number": {"type": "integer", "required": True},
                },
                "is_mutating": False,
                "requires_approval": False,
                "http_method": "GET",
            },
            {
                "operation": "list_pull_requests",
                "description": "List repository pull requests.",
                "parameters": {
                    "owner": {"type": "string", "required": True},
                    "repo": {"type": "string", "required": True},
                    "state": {"type": "string", "required": False, "default": "open"},
                },
                "is_mutating": False,
                "requires_approval": False,
                "http_method": "GET",
            },
        ]

    def validate_request(
        self, operation: str, params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate operation and parameters."""
        if not operation or not isinstance(operation, str):
            return False, "Operation name must be a non-empty string."

        # Reject write operations at the validation boundary
        if operation in WRITE_OPERATIONS:
            return False, (
                f"Write operation '{operation}' is explicitly rejected. "
                f"This adapter is read-only. Supported operations: {sorted(READ_ONLY_OPERATIONS)}"
            )

        if operation not in READ_ONLY_OPERATIONS:
            return False, (
                f"Unsupported operation '{operation}'. "
                f"Supported read-only operations: {sorted(READ_ONLY_OPERATIONS)}"
            )

        # Validate required parameters
        capabilities = {c["operation"]: c for c in self.discover_capabilities()}
        schema = capabilities.get(operation, {})
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
        """Execute a read-only GitHub API operation."""
        with self._lock:
            # 1. Validate operation
            valid, err = self.validate_request(operation, params)
            if not valid:
                return {"status": "error", "error": err}

            # 2. ENFORCE read-only at execution boundary
            if operation in WRITE_OPERATIONS or operation not in READ_ONLY_OPERATIONS:
                return {
                    "status": "error",
                    "error": f"Security enforcement: Operation '{operation}' rejected. Read-only access only.",
                }

            # 3. Check token availability
            token = self._get_token()
            if not token:
                return {
                    "status": "error",
                    "error": "GitHub integration not configured: GITHUB_TOKEN environment variable not set.",
                    "is_configured": False,
                }

            # 4. Build and execute the API request
            try:
                url, headers = self._build_request(operation, params, token)
                response_data = self._make_get_request(url, headers)
                return self.normalize_response(response_data)
            except HTTPError as e:
                return self._handle_http_error(e, operation)
            except URLError as e:
                return {
                    "status": "error",
                    "error": f"Network error during GitHub API call: Connection failed.",
                }
            except TimeoutError:
                return {
                    "status": "error",
                    "error": f"GitHub API request timed out after {self._timeout}s.",
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": "Invalid JSON response from GitHub API.",
                }
            except Exception as exc:
                # Sanitize error to prevent token leakage
                sanitized = str(exc)
                if token and token in sanitized:
                    sanitized = sanitized.replace(token, "[REDACTED]")
                return {"status": "error", "error": f"GitHub API error: {sanitized}"}

    def _build_request(
        self, operation: str, params: Dict[str, Any], token: str
    ) -> Tuple[str, Dict[str, str]]:
        """Build the URL and headers for a GitHub API request."""
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "AB-SMART-AI-Agent/1.0",
        }

        owner = params.get("owner", "")
        repo = params.get("repo", "")

        if operation == "get_authenticated_user":
            url = f"{GITHUB_API_BASE}/user"
        elif operation == "get_repository":
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
        elif operation == "list_issues":
            state = params.get("state", "open")
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues?state={state}"
        elif operation == "get_issue":
            issue_number = params.get("issue_number")
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}"
        elif operation == "list_pull_requests":
            state = params.get("state", "open")
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls?state={state}"
        else:
            raise ValueError(f"Unmapped operation: {operation}")

        return url, headers

    def _make_get_request(
        self, url: str, headers: Dict[str, str]
    ) -> Any:
        """Execute an HTTP GET request and parse JSON response."""
        req = Request(url, method="GET", headers=headers)
        import socket
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(self._timeout)
            with urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        finally:
            socket.setdefaulttimeout(old_timeout)

    def _handle_http_error(self, e: HTTPError, operation: str) -> Dict[str, Any]:
        """Handle HTTP errors without exposing sensitive information."""
        code = e.code
        if code == 401:
            return {
                "status": "error",
                "error": "GitHub API authentication failed: Invalid or expired token.",
                "http_status": 401,
            }
        elif code == 403:
            # Check for rate limiting
            retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") else None
            remaining = e.headers.get("X-RateLimit-Remaining") if hasattr(e, "headers") else None
            if remaining == "0" or (retry_after is not None):
                return {
                    "status": "error",
                    "error": "GitHub API rate limit exceeded. Please wait before retrying.",
                    "http_status": 403,
                    "rate_limited": True,
                }
            return {
                "status": "error",
                "error": "GitHub API access forbidden: Insufficient permissions.",
                "http_status": 403,
            }
        elif code == 404:
            return {
                "status": "error",
                "error": f"GitHub resource not found for operation '{operation}'.",
                "http_status": 404,
            }
        elif code == 429:
            return {
                "status": "error",
                "error": "GitHub API rate limit exceeded (HTTP 429). Please wait before retrying.",
                "http_status": 429,
                "rate_limited": True,
            }
        else:
            return {
                "status": "error",
                "error": f"GitHub API error: HTTP {code}.",
                "http_status": code,
            }

    def normalize_response(self, raw_response: Any) -> Dict[str, Any]:
        """Normalize raw GitHub API response into standard format."""
        return {
            "status": "success",
            "source_application": self.app.app_id,
            "is_mock": False,
            "data": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
