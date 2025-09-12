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
        self.llm = ChatGoogleGenerativeAI(
            model=model_config["model_name"],
            google_api_key=model_config["api_key"],
            temperature=0.1
        )
        self.prompt_template = self._create_prompt_template()
        self.structured_llm = self.llm.with_structured_output(LangGraphExpertResponse)
        self.chain = self.prompt_template | self.structured_llm
        logger.info(f"LangGraph Expert '{self.name}' initialized with model '{model_config['model_name']}'")

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

To generate your response, follow these steps:
1.  **Summarize the Problem**: Fill in the `problem_summary` field.
2.  **Reason Step-by-Step**: Fill in the `reasoning_steps` array, explaining how you will transform the prompt to meet all the criteria with a score of 0.96+.
3.  **Assess Confidence**: Fill in the `confidence_score` with a float between 0.0 and 1.0.
4.  **Provide the Final Solution**: Fill in the `solution` field with the new, high-quality prompt.
"""
        human_prompt = "Problem: {prompt}\n\nHistory: {history}"
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

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
            
            result = response.dict()
            result["improved_prompt"] = result.get("solution", original_prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error in LangGraphExpert '{self.name}': {e}")
            return {
                "error": str(e),
                "problem_summary": "Failed to analyze prompt due to an API error.",
                "reasoning_steps": [f"Error: {e}"],
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
