"""
core/process_workers.py

Multiprocessing worker entry points for cross-process isolation verification.
Placed in the core package so Windows multiprocessing child processes can reliably
import and unpickle targets regardless of how pytest launches.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Dict, Any

from core.orchestrator import AgentMessage, MessageType
from core.persistent_storage import SQLiteMessageStorage
from core.message_queue import MessagePriority


def worker_consume_approval(
    db_path: str,
    token: str,
    task_id: str,
    action: str,
    worker_id: int,
    result_queue: mp.Queue,
) -> None:
    """Worker function executed in an independent OS process for atomic HITL consumption."""
    try:
        storage = SQLiteMessageStorage(db_path=db_path)
        ok, err = storage.atomic_consume_approval(
            token=token,
            consumed_by=f"WorkerProcess_{worker_id}",
            task_id=task_id,
            action=action,
        )
        result_queue.put({"worker_id": worker_id, "success": ok, "error": err})
    except Exception as exc:
        result_queue.put({"worker_id": worker_id, "success": False, "error": str(exc)})


def worker_insert_message(
    db_path: str,
    message_dict: Dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """Worker function testing cross-process message deduplication."""
    try:
        storage = SQLiteMessageStorage(db_path=db_path)
        msg = AgentMessage(
            message_id=message_dict["message_id"],
            task_id=message_dict["task_id"],
            agent_name="WorkerProcess",
            sender="WorkerProcess",
            receiver="SupervisorAgent",
            stage="execution",
            status="completed",
            message_type=MessageType.TASK_RESULT,
            input_data="data",
            output_data="result",
        )
        ok = storage.insert_message(msg, priority=MessagePriority.NORMAL)
        result_queue.put({"success": ok})
    except Exception as exc:
        result_queue.put({"success": False, "error": str(exc)})
