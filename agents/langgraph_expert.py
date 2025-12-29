#!/us/bin/env python3
"""
LangGraph Expert Agent for Multi-Agent Prompt Engineering System

This module defines a sophisticated expert agent for the LangGraph workflow,
featuring dynamic tool selection, enhanced reasoning, and structured output.
"""

from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Optional
import os
import json
import re

# Add the parent directory to the path so we can import from config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_logger, get_model_config

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
        model_config = get_model_config(provider="google")
        self.model_name = model_config["model_name"]
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=model_config["api_key"],
            temperature=0.1
        )
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
        system_prompt = """
You are a world-class expert in {expertise}. Your role is to transform vague, high-level prompts into detailed, actionable, and high-quality prompts that will achieve a score of 0.96 or higher based on the evaluation criteria.

**Evaluation Criteria to Optimize For:**
- **Actionability (Score: 0.96+):** The prompt must be immediately usable and executable. Use strong action verbs.
- **Clarity (Score: 0.96+):** The prompt must be unambiguous and easy to understand. Avoid jargon where possible, or explain it.
- **Specificity (Score: 0.96+):** The prompt must contain enough detail to guide the desired output. Include constraints, desired formats, and examples.
- **Structure (Score: 0.96+):** The prompt should be well-organized with clear sections.
- **Completeness (Score: 0.96+):** The prompt should include all necessary context for the task.
- **Domain Alignment (Score: 0.96+):** The prompt should align with best practices for the given domain.

**Example of a High-Quality Transformation:**

*   **Vague Prompt:** "Analyze data with Python"
*   **High-Quality Solution (Score > 0.96):**
    "You are a data scientist. Your task is to perform an exploratory data analysis (EDA) on a given dataset using Python.

    **Dataset:** A CSV file will be provided with columns: 'Date', 'Sales', 'Region', 'Product_Category'.

    **Requirements:**
    1.  **Load the data:** Use the pandas library to load the CSV file into a DataFrame.
    2.  **Data Cleaning:** Handle any missing values and correct any obvious data entry errors.
    3.  **Descriptive Statistics:** Calculate and present summary statistics (mean, median, std, etc.) for the 'Sales' column.
    4.  **Sales Trends:** Analyze and visualize the sales trends over time. Create a line plot of sales by date.
    5.  **Regional Performance:** Analyze sales performance by region. Create a bar chart showing total sales for each region.
    6.  **Product Analysis:** Analyze sales by product category. Create a pie chart showing the proportion of sales for each category.

    **Output Format:**
    -   Provide a summary of your findings in a markdown format.
    -   Include all Python code used for the analysis.
    -   All plots should be clearly labeled.

    **Constraints:**
    -   Use the following Python libraries: pandas, matplotlib, seaborn.
    -   The analysis should be self-contained in a single script."

Your final output MUST be a JSON object with the following flat structure:
{{
    "problem_summary": "A brief summary of the problem",
    "reasoning_steps": ["A list of strings representing the step-by-step reasoning process"],
    "confidence_score": 0.98,
    "solution": "The final, complete solution or response that meets the high-quality standard shown in the example."
}}

**CRITICAL INSTRUCTION**: You MUST transform and improve the prompt. The `solution` field must contain a NEW, IMPROVED, and ENHANCED version that is significantly more detailed and actionable. DO NOT simply copy the original prompt - you must expand, clarify, and enhance it.

To generate your response, follow these steps:
1.  **Summarize the Problem**: Fill in the `problem_summary` field.
2.  **Reason Step-by-Step**: Fill in the `reasoning_steps` array, explaining how you will transform the prompt to meet all the criteria with a score of 0.96+.
3.  **Assess Confidence**: Fill in the `confidence_score` with a float between 0.0 and 1.0.
4.  **Provide the Final Solution**: Fill in the `solution` field with a NEW, IMPROVED, and ENHANCED prompt that is significantly better than the original. The solution must be different from the input prompt - add details, structure, examples, and clarity.
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
