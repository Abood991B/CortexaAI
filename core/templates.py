"""
Prompt Templates Library for CortexaAI.

Curated, domain-specific prompt templates with variable substitution,
tagging, and usage tracking.  Persisted via the SQLite database layer.

All default templates are sourced from Claude's official Prompt Library:
https://docs.anthropic.com/en/prompt-library
"""

import json
import uuid
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from config.config import get_logger
from core.database import db

logger = get_logger(__name__)


class TemplateEngine:
    """Manages prompt templates with variable substitution and persistence."""

    def __init__(self):
        self._seed_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new template and persist it."""
        template_id = db.save_template(data)
        logger.info(f"Created template {template_id}: {data.get('name')}")
        return {**data, "id": template_id}

    def create(self, name: str, domain: str, template_text: str,
               description: str = "", variables: Optional[List[str]] = None,
               is_public: bool = True, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Convenience: create a template from keyword args."""
        data = {
            "name": name,
            "domain": domain,
            "template": template_text,
            "description": description,
            "variables": variables or re.findall(r"\{\{(\w+)\}\}", template_text),
            "tags": [domain],
            "author": "user",
            "owner_id": owner_id,
            "is_public": is_public,
        }
        return self.create_template(data)

    def update_template(self, template_id: str, data: Dict[str, Any], user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update an existing user-created template. Only the owner can edit."""
        existing = db.get_template(template_id)
        if not existing:
            return None
        if existing.get("author", "system") == "system":
            return {"error": "Cannot edit system templates"}
        # Ownership check: only the owner can edit
        owner = existing.get("owner_id")
        if owner and user_id and owner != user_id:
            return {"error": "You do not have permission to edit this template"}

        updated = db.update_template(template_id, data)
        if updated:
            logger.info(f"Updated template {template_id}")
        return updated

    def delete_template(self, template_id: str, user_id: Optional[str] = None) -> dict:
        """Delete a user-created template. System templates cannot be deleted. Returns status dict."""
        existing = db.get_template(template_id)
        if not existing:
            return {"error": "not_found"}
        if existing.get("author", "system") == "system":
            return {"error": "Cannot delete system templates"}
        # Ownership check: only the owner can delete
        owner = existing.get("owner_id")
        if owner and user_id and owner != user_id:
            return {"error": "You do not have permission to delete this template"}

        db.delete_template(template_id)
        logger.info(f"Deleted template {template_id}")
        return {"status": "deleted"}

    def get_templates(self, domain: str = None, limit: int = 200, user_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve templates, optionally filtered by domain."""
        return db.get_templates(domain=domain, limit=limit, user_id=user_id)

    # Aliases used by the API layer
    list_all = get_templates

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single template by ID."""
        return db.get_template(template_id)

    # Alias
    get = get_template

    def render_template(self, template_id: str, variables: Dict[str, str]) -> Dict[str, Any]:
        """Render a template by substituting variables."""
        tpl = db.get_template(template_id)
        if not tpl:
            return {"error": f"Template {template_id} not found"}

        rendered = tpl["template"]
        for key, value in variables.items():
            rendered = rendered.replace("{{" + key + "}}", value)

        # Check for unresolved variables
        unresolved = re.findall(r"\{\{(\w+)\}\}", rendered)

        db.increment_template_usage(template_id)

        return {
            "template_id": template_id,
            "rendered_prompt": rendered,
            "variables_used": list(variables.keys()),
            "unresolved_variables": unresolved,
        }

    # Alias
    render = render_template

    def search_templates(self, query: str, domain: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """Search templates by keyword in name/description/tags."""
        all_templates = db.get_templates(domain=domain, limit=200, user_id=user_id)
        q = query.lower()
        results = []
        for tpl in all_templates:
            searchable = f"{tpl['name']} {tpl['description']} {' '.join(tpl.get('tags', []))}".lower()
            if q in searchable:
                results.append(tpl)
        return results

    # Alias
    search = search_templates

    # ------------------------------------------------------------------
    # Seed default templates
    # ------------------------------------------------------------------

    def _seed_defaults(self):
        """Seed built-in templates. Only manages system templates — never touches user templates."""
        existing = db.get_templates(limit=500)
        existing_names = {t["name"] for t in existing}

        defaults = self._get_default_templates()
        default_names = {t["name"] for t in defaults}

        # Only remove OLD system templates that are no longer in defaults.
        # NEVER delete user-created templates (author != 'system').
        with db.connection() as conn:
            for tpl in existing:
                if tpl.get("author", "system") == "system" and tpl["name"] not in default_names:
                    try:
                        conn.execute(
                            "DELETE FROM templates WHERE id = ? AND author = 'system'",
                            (tpl["id"],),
                        )
                    except Exception:
                        pass

        added = 0
        for tpl in defaults:
            if tpl["name"] not in existing_names:
                db.save_template(tpl)
                added += 1

        if added:
            logger.info(
                f"Seeded {added} new default templates "
                f"(total defaults: {len(defaults)})"
            )

    @staticmethod
    def _get_default_templates() -> List[Dict[str, Any]]:
        """
        Official Claude Prompt Library templates.
        Source: https://docs.anthropic.com/en/prompt-library
        """
        return [
            # =============================================================
            # CODING & DEVELOPMENT
            # =============================================================
            {
                "name": "Python Bug Buster",
                "domain": "coding",
                "description": (
                    "Detect and fix bugs in Python code. Provides corrected "
                    "code with explanations of what was wrong."
                ),
                "template": (
                    "Your task is to analyze the provided Python code snippet, "
                    "identify any bugs or errors present, and provide a "
                    "corrected version of the code that resolves these issues. "
                    "Explain the problems you found in the original code and "
                    "how your fixes address them. The corrected code should be "
                    "functional, efficient, and adhere to best practices in "
                    "Python programming.\n\n{{code}}"
                ),
                "variables": ["code"],
                "tags": [
                    "claude-library", "coding", "python",
                    "debugging", "bug-fix", "anthropic-official",
                ],
            },
            {
                "name": "Code Consultant",
                "domain": "coding",
                "description": (
                    "Optimize Python code for better performance. Identifies "
                    "inefficiencies and suggests improvements."
                ),
                "template": (
                    "Your task is to analyze the provided Python code snippet "
                    "and suggest improvements to optimize its performance. "
                    "Identify areas where the code can be made more efficient, "
                    "faster, or less resource-intensive. Provide specific "
                    "suggestions for optimization, along with explanations of "
                    "how these changes can enhance the code's performance. The "
                    "optimized code should maintain the same functionality as "
                    "the original code while demonstrating improved "
                    "efficiency.\n\n{{code}}"
                ),
                "variables": ["code"],
                "tags": [
                    "claude-library", "coding", "python",
                    "optimization", "performance", "anthropic-official",
                ],
            },
            {
                "name": "Code Clarifier",
                "domain": "coding",
                "description": (
                    "Explain code in plain language for non-technical "
                    "audiences using analogies and examples."
                ),
                "template": (
                    "Your task is to take the code snippet provided and "
                    "explain it in simple, easy-to-understand language. Break "
                    "down the code's functionality, purpose, and key "
                    "components. Use analogies, examples, and plain terms to "
                    "make the explanation accessible to someone with minimal "
                    "coding knowledge. Avoid using technical jargon unless "
                    "absolutely necessary, and provide clear explanations for "
                    "any jargon used. The goal is to help the reader "
                    "understand what the code does and how it works at a high "
                    "level.\n\n{{code}}"
                ),
                "variables": ["code"],
                "tags": [
                    "claude-library", "coding", "explanation",
                    "education", "anthropic-official",
                ],
            },
            {
                "name": "Function Fabricator",
                "domain": "coding",
                "description": (
                    "Create Python functions from natural language "
                    "descriptions, handling edge cases and validation."
                ),
                "template": (
                    "Your task is to create Python functions based on the "
                    "provided natural language requests. The requests will "
                    "describe the desired functionality of the function, "
                    "including the input parameters and expected return value. "
                    "Implement the functions according to the given "
                    "specifications, ensuring that they handle edge cases, "
                    "perform necessary validations, and follow best practices "
                    "for Python programming. Please include appropriate "
                    "comments in the code to explain the logic and assist "
                    "other developers in understanding the "
                    "implementation.\n\n{{request}}"
                ),
                "variables": ["request"],
                "tags": [
                    "claude-library", "coding", "python",
                    "functions", "generation", "anthropic-official",
                ],
            },
            {
                "name": "Efficiency Estimator",
                "domain": "coding",
                "description": (
                    "Calculate time complexity of algorithms using Big O "
                    "notation with step-by-step reasoning."
                ),
                "template": (
                    "Your task is to analyze the provided function or "
                    "algorithm and calculate its time complexity using Big O "
                    "notation. Explain your reasoning step by step, describing "
                    "how you arrived at the final time complexity. Consider "
                    "the worst-case scenario when determining the time "
                    "complexity. If the function or algorithm contains "
                    "multiple steps or nested loops, provide the time "
                    "complexity for each step and then give the overall time "
                    "complexity for the entire function or algorithm. Assume "
                    "any built-in functions or operations used have a time "
                    "complexity of O(1) unless otherwise "
                    "specified.\n\n{{code}}"
                ),
                "variables": ["code"],
                "tags": [
                    "claude-library", "coding", "algorithms",
                    "big-o", "complexity", "anthropic-official",
                ],
            },
            {
                "name": "Website Wizard",
                "domain": "coding",
                "description": (
                    "Create one-page websites from specifications with "
                    "embedded HTML, CSS, and JavaScript."
                ),
                "template": (
                    "Your task is to create a one-page website based on the "
                    "given specifications, delivered as an HTML file with "
                    "embedded JavaScript and CSS. The website should "
                    "incorporate a variety of engaging and interactive design "
                    "features, such as drop-down menus, dynamic text and "
                    "content, clickable buttons, and more. Ensure that the "
                    "design is visually appealing, responsive, and "
                    "user-friendly. The HTML, CSS, and JavaScript code should "
                    "be well-structured, efficiently organized, and properly "
                    "commented for readability and "
                    "maintainability.\n\n{{specifications}}"
                ),
                "variables": ["specifications"],
                "tags": [
                    "claude-library", "coding", "web",
                    "html", "css", "javascript", "anthropic-official",
                ],
            },
            {
                "name": "SQL Sorcerer",
                "domain": "coding",
                "description": (
                    "Transform natural language requests into valid SQL "
                    "queries with explanations."
                ),
                "template": (
                    "Transform the following natural language requests into "
                    "valid SQL queries. Assume a database with the following "
                    "tables and columns exists:\n\n"
                    "Customers:\n"
                    "- customer_id (INT, PRIMARY KEY)\n"
                    "- first_name (VARCHAR)\n"
                    "- last_name (VARCHAR)\n"
                    "- email (VARCHAR)\n"
                    "- phone (VARCHAR)\n"
                    "- address (VARCHAR)\n"
                    "- city (VARCHAR)\n"
                    "- state (VARCHAR)\n"
                    "- zip_code (VARCHAR)\n\n"
                    "Products:\n"
                    "- product_id (INT, PRIMARY KEY)\n"
                    "- product_name (VARCHAR)\n"
                    "- description (TEXT)\n"
                    "- category (VARCHAR)\n"
                    "- price (DECIMAL)\n"
                    "- stock_quantity (INT)\n\n"
                    "Orders:\n"
                    "- order_id (INT, PRIMARY KEY)\n"
                    "- customer_id (INT, FOREIGN KEY REFERENCES Customers)\n"
                    "- order_date (DATE)\n"
                    "- total_amount (DECIMAL)\n"
                    "- status (VARCHAR)\n\n"
                    "Order_Items:\n"
                    "- order_item_id (INT, PRIMARY KEY)\n"
                    "- order_id (INT, FOREIGN KEY REFERENCES Orders)\n"
                    "- product_id (INT, FOREIGN KEY REFERENCES Products)\n"
                    "- quantity (INT)\n"
                    "- price (DECIMAL)\n\n"
                    "Reviews:\n"
                    "- review_id (INT, PRIMARY KEY)\n"
                    "- product_id (INT, FOREIGN KEY REFERENCES Products)\n"
                    "- customer_id (INT, FOREIGN KEY REFERENCES Customers)\n"
                    "- rating (INT)\n"
                    "- comment (TEXT)\n"
                    "- review_date (DATE)\n\n"
                    "Provide the SQL query that would retrieve the data based "
                    "on the user's request, along with a brief explanation of "
                    "how the query works.\n\n{{request}}"
                ),
                "variables": ["request"],
                "tags": [
                    "claude-library", "coding", "sql",
                    "database", "queries", "anthropic-official",
                ],
            },

            # =============================================================
            # PRODUCTIVITY & BUSINESS
            # =============================================================
            {
                "name": "Excel Formula Expert",
                "domain": "productivity",
                "description": (
                    "Generate advanced Excel formulas for complex "
                    "calculations and data manipulations."
                ),
                "template": (
                    "As an Excel Formula Expert, your task is to provide "
                    "advanced Excel formulas that perform the complex "
                    "calculations or data manipulations described by the "
                    "user. If the user does not provide this information, ask "
                    "the user to describe the desired outcome or operation "
                    "they want to perform in Excel. Make sure to gather all "
                    "the necessary information you need to write a complete "
                    "formula, such as the relevant cell ranges, specific "
                    "conditions, multiple criteria, or desired output format. "
                    "Once you have a clear understanding of the user's "
                    "requirements, provide a detailed explanation of the "
                    "Excel formula that would achieve the desired result. "
                    "Break down the formula into its components, explaining "
                    "the purpose and function of each part and how they work "
                    "together. Additionally, provide any necessary context or "
                    "tips for using the formula effectively within an Excel "
                    "worksheet.\n\n{{request}}"
                ),
                "variables": ["request"],
                "tags": [
                    "claude-library", "productivity", "excel",
                    "formulas", "data", "anthropic-official",
                ],
            },
            {
                "name": "Meeting Scribe",
                "domain": "productivity",
                "description": (
                    "Transform meeting notes into concise summaries with "
                    "key takeaways and action items."
                ),
                "template": (
                    "Your task is to review the provided meeting notes and "
                    "create a concise summary that captures the essential "
                    "information, focusing on key takeaways and action items "
                    "assigned to specific individuals or departments during "
                    "the meeting. Use clear and professional language, and "
                    "organize the summary in a logical manner using "
                    "appropriate formatting such as headings, subheadings, "
                    "and bullet points. Ensure that the summary is easy to "
                    "understand and provides a comprehensive but succinct "
                    "overview of the meeting's content, with a particular "
                    "focus on clearly indicating who is responsible for each "
                    "action item.\n\n{{meeting_notes}}"
                ),
                "variables": ["meeting_notes"],
                "tags": [
                    "claude-library", "productivity", "meetings",
                    "summaries", "business", "anthropic-official",
                ],
            },
            {
                "name": "Data Organizer",
                "domain": "productivity",
                "description": (
                    "Convert unstructured text into well-organized JSON "
                    "tables with proper structure."
                ),
                "template": (
                    "Your task is to take the unstructured text provided and "
                    "convert it into a well-organized table format using "
                    "JSON. Identify the main entities, attributes, or "
                    "categories mentioned in the text and use them as keys "
                    "in the JSON object. Then, extract the relevant "
                    "information from the text and populate the "
                    "corresponding values in the JSON object. Ensure that "
                    "the data is accurately represented and properly "
                    "formatted within the JSON structure. The resulting JSON "
                    "table should provide a clear, structured overview of "
                    "the information presented in the original "
                    "text.\n\n{{text}}"
                ),
                "variables": ["text"],
                "tags": [
                    "claude-library", "productivity", "data",
                    "json", "organization", "anthropic-official",
                ],
            },

            # =============================================================
            # WRITING & EDITING
            # =============================================================
            {
                "name": "Prose Polisher",
                "domain": "writing",
                "description": (
                    "Advanced AI copyeditor that refines and improves "
                    "written content with detailed suggestions."
                ),
                "template": (
                    "You are an AI copyeditor with a keen eye for detail and "
                    "a deep understanding of language, style, and grammar. "
                    "Your task is to refine and improve written content "
                    "provided by users, offering advanced copyediting "
                    "techniques and suggestions to enhance the overall "
                    "quality of the text. When a user submits a piece of "
                    "writing, follow these steps:\n\n"
                    "1. Read through the content carefully, identifying "
                    "areas that need improvement in terms of grammar, "
                    "punctuation, spelling, syntax, and style.\n\n"
                    "2. Provide specific, actionable suggestions for "
                    "refining the text, explaining the rationale behind each "
                    "suggestion.\n\n"
                    "3. Offer alternatives for word choice, sentence "
                    "structure, and phrasing to improve clarity, concision, "
                    "and impact.\n\n"
                    "4. Ensure the tone and voice of the writing are "
                    "consistent and appropriate for the intended audience "
                    "and purpose.\n\n"
                    "5. Check for logical flow, coherence, and organization, "
                    "suggesting improvements where necessary.\n\n"
                    "6. Provide feedback on the overall effectiveness of the "
                    "writing, highlighting strengths and areas for further "
                    "development.\n\n"
                    "7. Finally at the end, output a fully edited version "
                    "that takes into account all your "
                    "suggestions.\n\n{{text}}"
                ),
                "variables": ["text"],
                "tags": [
                    "claude-library", "writing", "editing",
                    "copyediting", "grammar", "anthropic-official",
                ],
            },
            {
                "name": "Brand Builder",
                "domain": "writing",
                "description": (
                    "Create comprehensive brand identity design briefs "
                    "with name, logo, colors, and personality."
                ),
                "template": (
                    "Your task is to create a comprehensive design brief "
                    "for a holistic brand identity based on the given "
                    "specifications. The brand identity should encompass "
                    "various elements such as suggestions for the brand "
                    "name, logo, color palette, typography, visual style, "
                    "tone of voice, and overall brand personality. Ensure "
                    "that all elements work together harmoniously to create "
                    "a cohesive and memorable brand experience that "
                    "effectively communicates the brand's values, mission, "
                    "and unique selling proposition to its target audience. "
                    "Be detailed and comprehensive and provide enough "
                    "specific details for someone to create a truly unique "
                    "brand identity.\n\n{{brand_specs}}"
                ),
                "variables": ["brand_specs"],
                "tags": [
                    "claude-library", "writing", "branding",
                    "design", "marketing", "anthropic-official",
                ],
            },

            # =============================================================
            # RESEARCH & ANALYSIS
            # =============================================================
            {
                "name": "Cite Your Sources",
                "domain": "research",
                "description": (
                    "Answer questions about documents with properly cited "
                    "quotes and references."
                ),
                "template": (
                    "You are an expert research assistant. Here is a "
                    "document you will answer questions about:\n"
                    "<doc>\n{{document}}\n</doc>\n\n"
                    "First, find the quotes from the document that are most "
                    "relevant to answering the question, and then print "
                    "them in numbered order. Quotes should be relatively "
                    "short.\n\n"
                    "If there are no relevant quotes, write \"No relevant "
                    "quotes\" instead.\n\n"
                    "Then, answer the question, starting with \"Answer:\". "
                    "Do not include or reference quoted content verbatim in "
                    "the answer. Don't say \"According to Quote [1]\" when "
                    "answering. Instead make references to quotes relevant "
                    "to each section of the answer solely by adding their "
                    "bracketed numbers at the end of relevant "
                    "sentences.\n\n"
                    "Thus, the format of your overall response should look "
                    "like what's shown between the tags. Make sure to "
                    "follow the formatting and spacing exactly.\n\n"
                    "Quotes:\n"
                    "[1] \"Company X reported revenue of $12 million in "
                    "2021.\"\n"
                    "[2] \"Almost 90% of revenue came from widget sales, "
                    "with gadget sales making up the remainder.\"\n\n"
                    "Answer:\n"
                    "Company X earned $12 million. [1] Almost 90% of it "
                    "was from widget sales. [2]\n\n"
                    "Question: {{question}}"
                ),
                "variables": ["document", "question"],
                "tags": [
                    "claude-library", "research", "citations",
                    "analysis", "documents", "anthropic-official",
                ],
            },
            {
                "name": "Corporate Clairvoyant",
                "domain": "research",
                "description": (
                    "Analyze corporate reports to extract insights, "
                    "identify risks, and create executive memos."
                ),
                "template": (
                    "Your task is to analyze the following report:\n"
                    "<report>\n{{report}}\n</report>\n\n"
                    "Summarize this annual report in a concise and clear "
                    "manner, and identify key market trends and takeaways. "
                    "Output your findings as a short memo I can send to my "
                    "team. The goal of the memo is to ensure my team stays "
                    "up to date on how financial institutions are faring "
                    "and qualitatively forecast and identify whether there "
                    "are any operating and revenue risks to be expected in "
                    "the coming quarter. Make sure to include all relevant "
                    "details in your summary and analysis."
                ),
                "variables": ["report"],
                "tags": [
                    "claude-library", "research", "finance",
                    "analysis", "reports", "anthropic-official",
                ],
            },
            {
                "name": "Review Classifier",
                "domain": "research",
                "description": (
                    "Categorize user feedback into predefined categories "
                    "with sentiment analysis."
                ),
                "template": (
                    "You are an AI assistant trained to categorize user "
                    "feedback into predefined categories, along with "
                    "sentiment analysis for each category. Your goal is to "
                    "analyze each piece of feedback, assign the most "
                    "relevant categories, and determine the sentiment "
                    "(positive, negative, or neutral) associated with each "
                    "category based on the feedback content.\n\n"
                    "Predefined Categories:\n"
                    "- Product Features and Functionality\n"
                    "    - Core Features\n"
                    "    - Add-ons and Integrations\n"
                    "    - Customization and Configuration\n"
                    "- User Experience and Design\n"
                    "    - Ease of Use\n"
                    "    - Navigation and Discoverability\n"
                    "    - Visual Design and Aesthetics\n"
                    "    - Accessibility\n"
                    "- Performance and Reliability\n"
                    "    - Speed and Responsiveness\n"
                    "    - Uptime and Availability\n"
                    "    - Scalability\n"
                    "    - Bug Fixes and Error Handling\n"
                    "- Customer Support and Service\n"
                    "    - Responsiveness and Availability\n"
                    "    - Knowledge and Expertise\n"
                    "    - Issue Resolution and Follow-up\n"
                    "    - Self-Service Resources\n"
                    "- Billing, Pricing, and Licensing\n"
                    "    - Pricing Model and Tiers\n"
                    "    - Billing Processes and Invoicing\n"
                    "    - License Management\n"
                    "    - Upgrades and Renewals\n"
                    "- Security, Compliance, and Privacy\n"
                    "    - Data Protection and Confidentiality\n"
                    "    - Access Control and Authentication\n"
                    "    - Regulatory Compliance\n"
                    "    - Incident Response and Monitoring\n"
                    "- Mobile and Cross-Platform Compatibility\n"
                    "    - Mobile App Functionality\n"
                    "    - Synchronization and Data Consistency\n"
                    "    - Responsive Design\n"
                    "    - Device and OS Compatibility\n"
                    "- Third-Party Integrations and API\n"
                    "    - Integration Functionality and Reliability\n"
                    "    - API Documentation and Support\n"
                    "    - Customization and Extensibility\n"
                    "- Onboarding, Training, and Documentation\n"
                    "    - User Guides and Tutorials\n"
                    "    - Webinars and Workshops\n"
                    "    - Knowledge Base and FAQs\n"
                    "    - Community and Forums\n\n"
                    "{{feedback}}"
                ),
                "variables": ["feedback"],
                "tags": [
                    "claude-library", "research", "feedback",
                    "classification", "sentiment", "anthropic-official",
                ],
            },
            {
                "name": "Ethical Dilemma Navigator",
                "domain": "research",
                "description": (
                    "Navigate complex ethical dilemmas with multiple "
                    "frameworks and perspectives."
                ),
                "template": (
                    "Help the user navigate a complex ethical dilemma by "
                    "identifying core ethical principles, exploring "
                    "different ethical frameworks, considering potential "
                    "consequences, acknowledging complexity, encouraging "
                    "personal reflection, and offering additional "
                    "resources. Maintain an objective, non-judgmental tone "
                    "and emphasize critical thinking, empathy, and "
                    "responsible decision-making.\n\n{{dilemma}}"
                ),
                "variables": ["dilemma"],
                "tags": [
                    "claude-library", "research", "ethics",
                    "philosophy", "decision-making", "anthropic-official",
                ],
            },

            # =============================================================
            # EDUCATION
            # =============================================================
            {
                "name": "Lesson Planner",
                "domain": "education",
                "description": (
                    "Create comprehensive, engaging lesson plans for any "
                    "subject and grade level."
                ),
                "template": (
                    "Your task is to create a comprehensive, engaging, and "
                    "well-structured lesson plan on the given subject. The "
                    "lesson plan should be designed for a 60-minute class "
                    "session and should cater to a specific grade level or "
                    "age group. Begin by stating the lesson objectives, "
                    "which should be clear, measurable, and aligned with "
                    "relevant educational standards. Next, provide a "
                    "detailed outline of the lesson, breaking it down into "
                    "an introduction, main activities, and a conclusion. "
                    "For each section, describe the teaching methods, "
                    "learning activities, and resources you will use to "
                    "effectively convey the content and engage the "
                    "students. Finally, describe the assessment methods you "
                    "will employ to evaluate students' understanding and "
                    "mastery of the lesson objectives. The lesson plan "
                    "should be well-organized, easy to follow, and promote "
                    "active learning and critical "
                    "thinking.\n\n"
                    "Subject: {{subject}}\n"
                    "Grade Level: {{grade_level}}"
                ),
                "variables": ["subject", "grade_level"],
                "tags": [
                    "claude-library", "education", "teaching",
                    "lesson-plans", "anthropic-official",
                ],
            },
        ]


# Global template engine instance
template_engine = TemplateEngine()
