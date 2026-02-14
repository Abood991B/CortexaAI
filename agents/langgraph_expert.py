#!/usr/bin/env python3
"""
LangGraph Expert Agent for Multi-Agent Prompt Engineering System

This module defines a sophisticated expert agent for the LangGraph workflow,
featuring dynamic tool selection, enhanced reasoning, and structured output.

The agent supports both JSON-mode models (Gemini) and text-only models (Gemma)
with automatic fallback for maximum compatibility.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from typing import List, Dict, Any, Optional
import os
import json
import re

# Add the parent directory to the path so we can import from config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_logger, settings
from config.llm_providers import get_llm

logger = get_logger(__name__)

# --- Pydantic Models for Structured Output ---

class ToolSelection(BaseModel):
    """Represents a tool selected for a specific task."""
    tool_name: str = Field(..., description="The name of the selected tool")
    tool_input: Dict[str, Any] = Field(..., description="The input parameters for the tool")

class LangGraphExpertResponse(BaseModel):
    """Structured response from the LangGraph expert."""
    problem_summary: str = Field(..., description="A brief summary of the problem")
    reasoning_steps: List[str] = Field(..., description="Step-by-step reasoning process")
    confidence_score: float = Field(..., description="Confidence score (0.0 to 1.0)")
    solution: str = Field(..., description="The final solution or response")

# --- LangGraph Expert Agent ---

class LangGraphExpert:
    """
    An advanced expert agent for the LangGraph workflow.
    
    This expert uses structured output and a more complex reasoning process
    to provide high-quality, detailed responses.
    """

    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.model_name = settings.default_model_name
        self.llm = get_llm(temperature=0.1)
        # Check if model supports system instructions and JSON mode
        # Models that don't support system instructions: gemma-*-27b-*, gemma-*-2b-*, etc.
        # Models that don't support JSON mode: gemma-*-27b-*, gemma-*-2b-*, etc.
        # Models that do support both: gemini-*, gemini-2.0-*, gemini-1.5-*, etc.
        model_lower = self.model_name.lower()
        
        # Check for Gemma models (they have limitations)
        is_gemma_model = "gemma" in model_lower
        # Gemma 27B and 2B models have limitations
        is_limited_gemma = is_gemma_model and ("27b" in model_lower or "2b" in model_lower)
        
        # Check for Gemini models (they support everything)
        is_gemini_model = "gemini" in model_lower
        
        # Determine capabilities
        # Gemini models support both features
        if is_gemini_model:
            self.supports_system_instructions = True
            self.supports_json_mode = True
        # Limited Gemma models support neither
        elif is_limited_gemma:
            self.supports_system_instructions = False
            self.supports_json_mode = False
        # Other Gemma models or unknown models - assume they support features (will fallback on error)
        else:
            self.supports_system_instructions = True
            self.supports_json_mode = True
        
        self.prompt_template = self._create_prompt_template()
        
        # Create chain based on model capabilities
        if self.supports_json_mode:
            self.structured_llm = self.llm.with_structured_output(LangGraphExpertResponse)
            self.chain = self.prompt_template | self.structured_llm
        else:
            # For models without JSON mode, use regular LLM and parse JSON manually
            self.chain = self.prompt_template | self.llm
        
        logger.info(f"LangGraph Expert '{self.name}' initialized with model '{self.model_name}' (system instructions: {self.supports_system_instructions}, JSON mode: {self.supports_json_mode})")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Creates the prompt template for the expert agent."""
        system_prompt = """You are a world-class prompt engineer specialising in **{expertise}**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transform the user's prompt into a **production-ready, high-quality** prompt that
scores ≥ 0.96 on every evaluation criterion below. You MUST substantially
improve the original — never return it unchanged.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVALUATION CRITERIA (target ≥ 0.96 each)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Actionability** – immediately usable; strong action verbs; no ambiguity
   about *what* to do.
2. **Clarity** – crystal-clear language; jargon explained or replaced.
3. **Specificity** – constraints, desired format, examples, edge-cases
   explicitly stated.
4. **Structure** – logical sections with headings/numbered lists; easy to scan.
5. **Completeness** – all context needed to fulfil the task is present;
   no implicit assumptions left unstated.
6. **Domain Alignment** – reflects best practices of **{expertise}**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TRANSFORMATION STEPS (follow in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **Summarise** the user's intent in 1-2 sentences (→ `problem_summary`).
2. **Reason step-by-step** about what the prompt lacks measured against every
   criterion above (→ `reasoning_steps`).
3. **Assess your confidence** that the rewritten prompt will score ≥ 0.96
   (→ `confidence_score`, float 0.0 – 1.0).
4. **Produce the final solution** — a *completely rewritten* prompt that is
   significantly more detailed, structured, and actionable than the input
   (→ `solution`).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Input prompt:** "Analyze data with Python"

**High-quality solution (score > 0.96):**
> You are a data scientist. Perform an exploratory data analysis (EDA) on a
> CSV dataset with columns: Date, Sales, Region, Product_Category.
>
> **Requirements:**
> 1. Load data with pandas; validate schema on load.
> 2. Clean missing values (document strategy chosen).
> 3. Compute descriptive statistics for Sales.
> 4. Visualise sales trends over time (line plot) and by region (bar chart).
> 5. Analyse product-category proportions (pie chart).
>
> **Output format:** Markdown report with embedded code and labelled plots.
> **Libraries:** pandas, matplotlib, seaborn.
> **Constraint:** Single self-contained script; set random seed = 42.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT FORMAT (strict JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return **only** a JSON object — no markdown fences, no commentary:
{{
    "problem_summary": "<string>",
    "reasoning_steps": ["<step 1>", "<step 2>", "..."],
    "confidence_score": <float 0.0-1.0>,
    "solution": "<the fully rewritten prompt>"
}}

⚠️ The `solution` field MUST be a **new, improved, and enhanced** version.
   Copying the original verbatim is a failure case.
"""
        
        if self.supports_system_instructions:
            # Use system message for models that support it
            human_prompt = "Problem: {prompt}\n\nHistory: {history}"
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
        else:
            # Merge system prompt into human message for models that don't support system instructions
            human_prompt = f"{system_prompt}\n\nProblem: {{prompt}}\n\nHistory: {{history}}"
            return ChatPromptTemplate.from_messages([
                ("human", human_prompt)
            ])
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from text when JSON mode is not supported."""
        try:
            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object boundaries
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx + 1]
                else:
                    # If no JSON found, treat entire response as solution
                    return {
                        "problem_summary": "Prompt improvement completed",
                        "reasoning_steps": ["Analyzed and improved the prompt"],
                        "confidence_score": 0.85,
                        "solution": text.strip()
                    }
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
            
            # Ensure all required fields exist
            result = {
                "problem_summary": parsed.get("problem_summary", "Prompt analysis completed"),
                "reasoning_steps": parsed.get("reasoning_steps", ["Analyzed the prompt"]),
                "confidence_score": float(parsed.get("confidence_score", 0.85)),
                "solution": parsed.get("solution", text.strip())
            }
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}. Attempting to extract solution from text.")
            # Try to find if there's an improved prompt in the text even if JSON parsing failed
            # Look for patterns like "solution:", "improved prompt:", etc.
            solution_patterns = [
                r'"solution"\s*:\s*"([^"]+)"',
                r'"solution"\s*:\s*\'([^\']+)\'',
                r'solution["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                r'improved\s+prompt["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            ]
            
            extracted_solution = None
            for pattern in solution_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_solution = match.group(1).strip()
                    logger.info(f"Extracted solution from text using pattern")
                    break
            
            # If no pattern match, use the entire text but log a warning
            if not extracted_solution:
                logger.warning(f"Could not extract structured solution from response. Using full text as solution.")
                extracted_solution = text.strip()
            
            return {
                "problem_summary": "Prompt improvement completed (parsed from text)",
                "reasoning_steps": ["Analyzed and improved the prompt"],
                "confidence_score": 0.85,
                "solution": extracted_solution
            }

    async def improve_prompt(self, original_prompt: str, prompt_type: str = "raw", key_topics: Optional[List[str]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a prompt with enhanced reasoning and structured output.
        
        Args:
            original_prompt: The user's prompt.
            prompt_type: The type of prompt (raw, structured).
            key_topics: A list of key topics.
            history: A list of previous interactions.
            
        Returns:
            A dictionary containing the structured response.
        """
        logger.info(f"LangGraph Expert '{self.name}' processing prompt: {original_prompt[:100]}...")
        
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (history or [])])
        
        try:
            response = await self.chain.ainvoke({
                "expertise": self.expertise,
                "prompt": original_prompt,
                "history": history_str
            })
            
            # Handle response based on whether we're using structured output or not
            if self.supports_json_mode:
                # Structured output - response is already a Pydantic model
                result = response.dict()
            else:
                # Regular output - need to parse JSON from text
                response_text = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"Raw response from LLM (first 1000 chars): {response_text[:1000]}")
                result = self._parse_json_response(response_text)
                logger.info(f"Parsed result keys: {list(result.keys())}, solution length: {len(result.get('solution', ''))}")
            
            improved_prompt = result.get("solution", original_prompt)
            result["improved_prompt"] = improved_prompt
            
            # Log improvement details for debugging
            if improved_prompt != original_prompt:
                logger.info(f"LangGraph Expert '{self.name}' successfully improved prompt (length: {len(original_prompt)} -> {len(improved_prompt)} chars)")
                logger.debug(f"Original: {original_prompt[:100]}...")
                logger.debug(f"Improved: {improved_prompt[:100]}...")
            else:
                logger.warning(f"LangGraph Expert '{self.name}' returned same prompt as input - improvement may have failed or model returned original")
            
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error in LangGraphExpert '{self.name}': {e}")
            
            # Check for specific error types and use fallback
            if ("developer instruction" in error_str.lower() or 
                "system instruction" in error_str.lower() or
                "json mode" in error_str.lower()):
                logger.warning(f"Model '{self.model_name}' doesn't support required feature. Using fallback method.")
                # Try again without structured output and with merged prompt
                try:
                    logger.info("Attempting fallback: using text-based prompt without structured output...")
                    # Get the prompt content (handle both system+human and human-only formats)
                    if len(self.prompt_template.messages) > 1:
                        # Has system message - merge it
                        system_content = self.prompt_template.messages[0].content.format(expertise=self.expertise) if hasattr(self.prompt_template.messages[0], 'content') else ""
                        human_content = self.prompt_template.messages[1].content
                        merged_prompt = f"{system_content}\n\n{human_content}".format(
                            expertise=self.expertise,
                            prompt=original_prompt,
                            history=history_str
                        )
                    else:
                        # Already merged format
                        merged_prompt = self.prompt_template.messages[0].content.format(
                            expertise=self.expertise,
                            prompt=original_prompt,
                            history=history_str
                        )
                    
                    simple_chain = ChatPromptTemplate.from_messages([("human", merged_prompt)]) | self.llm
                    response = await simple_chain.ainvoke({})
                    # Extract and parse response
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    result = self._parse_json_response(response_text)
                    result["improved_prompt"] = result.get("solution", original_prompt)
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            return {
                "error": error_str,
                "problem_summary": "Failed to analyze prompt due to an API error.",
                "reasoning_steps": [f"Error: {error_str}"],
                "confidence_score": 0.0,
                "solution": original_prompt,
                "improved_prompt": original_prompt
            }

# --- Factory Function ---

def get_langgraph_expert(name: str, expertise: str) -> LangGraphExpert:
    """Factory function to create a LangGraphExpert instance."""
    return LangGraphExpert(name=name, expertise=expertise)

# --- Example Usage ---

async def main():
    """Example of how to use the LangGraphExpert."""
    expert = get_langgraph_expert(
        name="Software Engineering Expert",
        expertise="Software Engineering and Python"
    )
    
    prompt = "Design a Python function to perform asynchronous API calls and handle rate limiting."
    
    result = await expert.improve_prompt(original_prompt=prompt)
    
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # This allows for testing the expert independently
    import asyncio
    asyncio.run(main())
