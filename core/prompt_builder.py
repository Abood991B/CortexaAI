"""
Visual Prompt Builder API for CortexaAI.

Structured prompt assembly using composable blocks:
  Role → Context → Task → Constraints → Output Format → Examples

Each block can be independently edited, reordered and serialised.
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional
from enum import Enum

from config.config import get_logger

logger = get_logger(__name__)


class BlockType(str, Enum):
    ROLE = "role"
    CONTEXT = "context"
    TASK = "task"
    CONSTRAINTS = "constraints"
    OUTPUT_FORMAT = "output_format"
    EXAMPLES = "examples"
    CUSTOM = "custom"


# Default ordering weight
_BLOCK_ORDER = {
    BlockType.ROLE: 0,
    BlockType.CONTEXT: 1,
    BlockType.TASK: 2,
    BlockType.CONSTRAINTS: 3,
    BlockType.OUTPUT_FORMAT: 4,
    BlockType.EXAMPLES: 5,
    BlockType.CUSTOM: 6,
}


class PromptBlock:
    """A single building block of a prompt."""

    def __init__(
        self,
        block_type: BlockType,
        content: str,
        label: Optional[str] = None,
        order: Optional[int] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.block_type = block_type
        self.content = content
        self.label = label or block_type.value.replace("_", " ").title()
        self.order = order if order is not None else _BLOCK_ORDER.get(block_type, 99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "block_type": self.block_type.value,
            "content": self.content,
            "label": self.label,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptBlock":
        block = cls(
            block_type=BlockType(data["block_type"]),
            content=data["content"],
            label=data.get("label"),
            order=data.get("order"),
        )
        if "id" in data:
            block.id = data["id"]
        return block


# ── Prompt Builder Presets per Domain ────────────────────────────────────
_PRESETS: Dict[str, List[Dict[str, str]]] = {
    "software_engineering": [
        {"type": "role", "content": "You are a senior software engineer with expertise in system design and clean code."},
        {"type": "context", "content": "The project uses {{language}} and follows {{framework}} conventions."},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "constraints", "content": "- Follow SOLID principles\n- Include error handling\n- Add type hints"},
        {"type": "output_format", "content": "Provide the code with inline comments explaining key decisions."},
    ],
    "data_science": [
        {"type": "role", "content": "You are a data scientist specialising in {{specialty}}."},
        {"type": "context", "content": "Dataset: {{dataset_description}}"},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "constraints", "content": "- Explain statistical assumptions\n- Note potential biases"},
        {"type": "output_format", "content": "Present findings with methodology, results, and interpretation sections."},
    ],
    "report_writing": [
        {"type": "role", "content": "You are a professional report writer."},
        {"type": "context", "content": "Audience: {{audience}}. Purpose: {{purpose}}."},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "output_format", "content": "Use executive summary, findings, and recommendations sections."},
    ],
    "education": [
        {"type": "role", "content": "You are an experienced educator in {{subject}}."},
        {"type": "context", "content": "Student level: {{level}}. Learning goals: {{goals}}."},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "output_format", "content": "Structure with clear learning objectives, content, and assessment."},
    ],
    "business_strategy": [
        {"type": "role", "content": "You are a business strategy consultant."},
        {"type": "context", "content": "Industry: {{industry}}. Company stage: {{stage}}."},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "constraints", "content": "- Base recommendations on data\n- Consider market trends"},
        {"type": "output_format", "content": "Provide analysis with actionable recommendations and KPIs."},
    ],
    "creative_writing": [
        {"type": "role", "content": "You are a creative writer skilled in {{genre}}."},
        {"type": "context", "content": "Tone: {{tone}}. Target audience: {{audience}}."},
        {"type": "task", "content": "{{task_description}}"},
        {"type": "output_format", "content": "Write with vivid prose, strong voice and engaging narrative."},
    ],
}


class PromptBuilder:
    """Build prompts visually from structured blocks."""

    def __init__(self):
        self._sessions: Dict[str, List[PromptBlock]] = {}

    # ── Session management ───────────────────────────────────────────────
    def create_session(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Create a new builder session, optionally pre-loaded with a preset."""
        session_id = str(uuid.uuid4())[:10]
        blocks: List[PromptBlock] = []

        if domain and domain in _PRESETS:
            for preset in _PRESETS[domain]:
                blocks.append(
                    PromptBlock(
                        block_type=BlockType(preset["type"]),
                        content=preset["content"],
                    )
                )

        self._sessions[session_id] = blocks
        return {
            "session_id": session_id,
            "domain": domain,
            "blocks": [b.to_dict() for b in blocks],
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return None
        return {
            "session_id": session_id,
            "blocks": [b.to_dict() for b in blocks],
        }

    # ── Block operations ─────────────────────────────────────────────────
    def add_block(
        self,
        session_id: str,
        block_type: str,
        content: str,
        label: Optional[str] = None,
        order: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return None
        block = PromptBlock(
            block_type=BlockType(block_type),
            content=content,
            label=label,
            order=order,
        )
        blocks.append(block)
        return block.to_dict()

    def update_block(
        self, session_id: str, block_id: str, content: str
    ) -> Optional[Dict[str, Any]]:
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return None
        for b in blocks:
            if b.id == block_id:
                b.content = content
                return b.to_dict()
        return None

    def remove_block(self, session_id: str, block_id: str) -> bool:
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return False
        self._sessions[session_id] = [b for b in blocks if b.id != block_id]
        return True

    def reorder_blocks(
        self, session_id: str, block_ids: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return None
        id_map = {b.id: b for b in blocks}
        reordered = []
        for i, bid in enumerate(block_ids):
            if bid in id_map:
                id_map[bid].order = i
                reordered.append(id_map[bid])
        # Append any blocks not in the provided list
        seen = set(block_ids)
        for b in blocks:
            if b.id not in seen:
                b.order = len(reordered)
                reordered.append(b)
        self._sessions[session_id] = reordered
        return [b.to_dict() for b in reordered]

    # ── Assemble ─────────────────────────────────────────────────────────
    def assemble(
        self,
        session_id: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assemble blocks into a final prompt string."""
        blocks = self._sessions.get(session_id)
        if blocks is None:
            return None

        sorted_blocks = sorted(blocks, key=lambda b: b.order)
        parts: List[str] = []
        for b in sorted_blocks:
            content = b.content
            # Variable substitution
            if variables:
                for var, val in variables.items():
                    content = content.replace(f"{{{{{var}}}}}", val)
            if b.block_type == BlockType.ROLE:
                parts.append(f"**Role:** {content}")
            elif b.block_type == BlockType.CONTEXT:
                parts.append(f"**Context:** {content}")
            elif b.block_type == BlockType.TASK:
                parts.append(f"**Task:** {content}")
            elif b.block_type == BlockType.CONSTRAINTS:
                parts.append(f"**Constraints:**\n{content}")
            elif b.block_type == BlockType.OUTPUT_FORMAT:
                parts.append(f"**Output Format:** {content}")
            elif b.block_type == BlockType.EXAMPLES:
                parts.append(f"**Examples:**\n{content}")
            else:
                parts.append(content)

        assembled = "\n\n".join(parts)
        return {
            "prompt": assembled,
            "block_count": len(sorted_blocks),
            "char_count": len(assembled),
            "variables_used": list(variables.keys()) if variables else [],
        }

    # ── Presets ──────────────────────────────────────────────────────────
    def list_presets(self) -> List[str]:
        return list(_PRESETS.keys())

    def get_preset(self, domain: str) -> Optional[List[Dict[str, str]]]:
        return _PRESETS.get(domain)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Global instance
prompt_builder = PromptBuilder()
