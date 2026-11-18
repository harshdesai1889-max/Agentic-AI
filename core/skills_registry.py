"""
core/skills_registry.py

Production Skills Architecture for Day 231 (Part 3).

Implements a lightweight, secure skills framework providing reusable instructions,
workflows, and supporting resources that help agents perform specialized tasks.

A Skill is NOT an agent, tool, or MCP server. A Skill is a structured guidance module
loaded selectively by an agent based on task context.

Implements:
1. Skill: Dataclass holding metadata, workflow instructions, references, and permissions.
2. SkillRegistry: Thread-safe repository discovering and loading skills from `skills/`.
3. Selective Skill Discovery: Matches task requirements to relevant skills, minimizing prompt context bloat.
4. Skill Security: Scans skill content for prompt injection before loading into agent contexts.
   Enforces that a skill never grants permissions beyond the executing agent's own role.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from utils.logging_config import get_logger

logger = get_logger()


@dataclass
class Skill:
    """Reusable skill definition encapsulating instructions, schemas, and references."""
    name: str
    description: str
    instructions: str
    version: str = "1.0.0"
    required_capabilities: List[str] = field(default_factory=list)
    input_requirements: Dict[str, Any] = field(default_factory=dict)
    output_format: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)
    allowed_roles: List[str] = field(default_factory=lambda: ["ResearcherAgent", "PlannerAgent", "ExecutorAgent", "SupervisorAgent"])
    tags: List[str] = field(default_factory=list)
    skill_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "required_capabilities": self.required_capabilities,
            "input_requirements": self.input_requirements,
            "output_format": self.output_format,
            "references": list(self.references.keys()),
            "allowed_roles": self.allowed_roles,
            "tags": self.tags,
            "skill_path": self.skill_path,
        }


class SkillRegistry:
    """Thread-safe registry managing the discovery, validation, and selective retrieval of skills."""

    _instance: Optional["SkillRegistry"] = None
    _lock = threading.RLock()

    def __init__(self, skills_root: Optional[Union[str, Path]] = None):
        self.skills_root = Path(skills_root) if skills_root else Path("c:/AgenticAI/day223_agents/skills")
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.RLock()
        self.scan_and_load_skills()

    @classmethod
    def get_instance(cls, skills_root: Optional[Union[str, Path]] = None) -> "SkillRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(skills_root=skills_root)
            return cls._instance

    def scan_and_load_skills(self) -> int:
        """Scan skills directory and parse all available SKILL.md bundles."""
        with self._lock:
            if not self.skills_root.exists():
                logger.warning(f"[SkillRegistry] Skills root directory not found: {self.skills_root}")
                return 0

            count = 0
            for skill_dir in self.skills_root.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        try:
                            skill = self._parse_skill_markdown(skill_file)
                            if skill:
                                self._skills[skill.name] = skill
                                count += 1
                        except Exception as exc:
                            logger.error(f"[SkillRegistry] Failed to parse skill at '{skill_file}': {exc}")

            logger.info(f"[SkillRegistry] Discovered and registered {count} skills from {self.skills_root}")
            return count

    def register_skill(self, skill: Skill) -> None:
        """Register a Skill object directly."""
        with self._lock:
            self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """Fetch a registered skill by exact name."""
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List summary metadata for all registered skills."""
        with self._lock:
            return [s.to_dict() for s in self._skills.values()]

    def discover_skills_for_task(
        self,
        task_description: str,
        agent_role: Optional[str] = None,
        max_results: int = 3,
        security_manager: Optional[Any] = None,
    ) -> List[Skill]:
        """Selectively discover and rank relevant skills based on task context and agent role.
        
        Avoids loading all skills into the prompt, selecting only highly relevant skills.
        Validates skill content for security before returning.
        """
        with self._lock:
            if not task_description:
                return []

            task_lower = task_description.lower()
            role_norm = (agent_role or "").strip().lower()

            scored_skills: List[Tuple[float, Skill]] = []

            for skill in self._skills.values():
                # 1. Role permission check
                if role_norm and skill.allowed_roles:
                    allowed_lower = [r.lower() for r in skill.allowed_roles]
                    if role_norm not in allowed_lower and not any(role_norm in r for r in allowed_lower):
                        continue

                # 2. Keyword relevance scoring
                score = 0.0
                name_clean = skill.name.lower().replace("-", " ").replace("_", " ")
                if name_clean in task_lower:
                    score += 5.0
                for word in name_clean.split():
                    if len(word) > 2 and word in task_lower:
                        score += 1.5

                desc_lower = skill.description.lower()
                for kw in ["review", "code", "python", "research", "analysis", "plan", "planning", "decompose"]:
                    if kw in task_lower and kw in desc_lower:
                        score += 1.0

                for tag in skill.tags:
                    if tag.lower() in task_lower:
                        score += 2.0

                if score > 0.0:
                    scored_skills.append((score, skill))

            # Sort descending by relevance score
            scored_skills.sort(key=lambda x: x[0], reverse=True)
            top_skills = [item[1] for item in scored_skills[:max_results]]

            # 3. Security verification: Ensure skill instructions do not contain malicious overrides
            sec = security_manager
            if sec is None:
                try:
                    from core.security import get_active_security_manager
                    sec = get_active_security_manager()
                except Exception:
                    sec = None

            safe_skills = []
            for s in top_skills:
                if sec is not None and hasattr(sec, "classify_input"):
                    dec, findings = sec.classify_input(s.instructions, source="skill_definition")
                    if dec == "deny" or getattr(dec, "value", None) == "deny":
                        logger.warning(f"[SkillRegistry] Skill '{s.name}' rejected by security policy: {findings}")
                        continue
                safe_skills.append(s)

            return safe_skills

    def _parse_skill_markdown(self, skill_file: Path) -> Optional[Skill]:
        """Parse YAML frontmatter and body sections from a SKILL.md file."""
        text = skill_file.read_text(encoding="utf-8")
        frontmatter = {}
        body = text

        # Check for YAML frontmatter between --- and ---
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            body = fm_match.group(2)
            # Lightweight YAML key-value parser to avoid external PyYAML dependency
            current_key = None
            for line in fm_text.splitlines():
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if ":" in line_str:
                    parts = line_str.split(":", 1)
                    k = parts[0].strip()
                    v = parts[1].strip().strip('"').strip("'")
                    if v == ">-":
                        current_key = k
                        frontmatter[k] = ""
                    else:
                        frontmatter[k] = v
                        current_key = None
                elif current_key and line.startswith("  "):
                    frontmatter[current_key] = (frontmatter[current_key] + " " + line_str).strip()

        skill_dir = skill_file.parent
        name = frontmatter.get("name") or skill_dir.name
        description = frontmatter.get("description", "").strip()

        # Load references from references/ subdirectory if present
        refs = {}
        refs_dir = skill_dir / "references"
        if refs_dir.exists() and refs_dir.is_dir():
            for ref_file in refs_dir.iterdir():
                if ref_file.is_file() and ref_file.suffix in (".md", ".txt"):
                    try:
                        refs[ref_file.name] = ref_file.read_text(encoding="utf-8")
                    except Exception:
                        pass

        # Identify tags and capabilities
        tags = []
        name_lower = name.lower()
        if "review" in name_lower or "code" in name_lower:
            tags.extend(["code_review", "python", "software_engineering"])
        if "research" in name_lower:
            tags.extend(["research", "evidence", "analysis"])
        if "plan" in name_lower:
            tags.extend(["planning", "decomposition", "scheduling"])

        return Skill(
            name=name,
            description=description or f"Skill guidance for {name}",
            instructions=body.strip(),
            version=frontmatter.get("version", "1.0.0"),
            references=refs,
            tags=tags,
            skill_path=str(skill_file),
        )
