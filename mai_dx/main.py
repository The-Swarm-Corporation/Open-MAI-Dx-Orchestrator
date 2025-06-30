"""
MAI Diagnostic Orchestrator (MAI-DxO)

This script provides a complete implementation of the "Sequential Diagnosis with Language Models"
paper, using the `swarms` framework. It simulates a virtual panel of physician-agents to perform
iterative medical diagnosis with cost-effectiveness optimization.

Based on the paper: "Sequential Diagnosis with Language Models"
(arXiv:2506.22405v1) by Nori et al.

Key Features:
- Virtual physician panel with specialized roles (Hypothesis, Test-Chooser, Challenger, Stewardship, Checklist)
- Multiple operational modes (instant, question_only, budgeted, no_budget, ensemble)
- Comprehensive cost tracking and budget management
- Clinical accuracy evaluation with 5-point Likert scale
- Gatekeeper system for realistic clinical information disclosure
- Ensemble methods for improved diagnostic accuracy

Example Usage:
    # Standard MAI-DxO usage
    orchestrator = MaiDxOrchestrator(model_name="gemini/gemini-2.5-flash")
    result = orchestrator.run(initial_case_info, full_case_details, ground_truth)

    # Budget-constrained variant
    budgeted_orchestrator = MaiDxOrchestrator.create_variant("budgeted", budget=5000)

    # Ensemble approach
    ensemble_result = orchestrator.run_ensemble(initial_case_info, full_case_details, ground_truth)
"""

# Enable debug mode if environment variable is set
import os
import json
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union, Literal

from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, Conversation

# Configure Loguru with beautiful formatting and features
logger.remove()  # Remove default handler

# Console handler with beautiful colors
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)


if os.getenv("MAIDX_DEBUG", "").lower() in ("1", "true", "yes"):
    logger.add(
        "logs/maidx_debug_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="3 days",
    )
    logger.info(
        "🐛 Debug logging enabled - logs will be written to logs/ directory"
    )

# File handler for persistent logging (optional - uncomment if needed)
# logger.add(
#     "logs/mai_dxo_{time:YYYY-MM-DD}.log",
#     rotation="1 day",
#     retention="7 days",
#     level="DEBUG",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
#     compression="zip"
# )

# --- Data Structures and Enums ---


class AgentRole(Enum):
    """Enumeration of roles for the virtual physician panel."""

    HYPOTHESIS = "Dr. Hypothesis"
    TEST_CHOOSER = "Dr. Test-Chooser"
    CHALLENGER = "Dr. Challenger"
    STEWARDSHIP = "Dr. Stewardship"
    CHECKLIST = "Dr. Checklist"
    CONSENSUS = "Consensus Coordinator"
    GATEKEEPER = "Gatekeeper"
    JUDGE = "Judge"


@dataclass
class DiagnosisResult:
    """Stores the final result of a diagnostic session."""

    final_diagnosis: str
    ground_truth: str
    accuracy_score: float
    accuracy_reasoning: str
    total_cost: int
    iterations: int
    conversation_history: str


class Action(BaseModel):
    """Pydantic model for a structured action decided by the consensus agent."""

    action_type: Literal["ask", "test", "diagnose"] = Field(
        ..., description="The type of action to perform."
    )
    content: Union[str, List[str]] = Field(
        ...,
        description="The content of the action (question, test name, or diagnosis).",
    )
    reasoning: str = Field(
        ..., description="The reasoning behind choosing this action."
    )


# --- Main Orchestrator Class ---


class MaiDxOrchestrator:
    """
    Implements the MAI Diagnostic Orchestrator (MAI-DxO) framework.
    This class orchestrates a virtual panel of AI agents to perform sequential medical diagnosis,
    evaluates the final diagnosis, and tracks costs.
    """

    def __init__(
        self,
        model_name: str = "gemini/gemini-2.5-flash",
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
        physician_visit_cost: int = 300,
        enable_budget_tracking: bool = False,
    ):
        """
        Initializes the MAI-DxO system.

        Args:
            model_name (str): The language model to be used by all agents.
            max_iterations (int): The maximum number of diagnostic loops.
            initial_budget (int): The starting budget for diagnostic tests.
            mode (str): The operational mode of MAI-DxO.
            physician_visit_cost (int): Cost per physician visit.
            enable_budget_tracking (bool): Whether to enable budget tracking.
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.enable_budget_tracking = enable_budget_tracking

        self.cumulative_cost = 0
        self.differential_diagnosis = "Not yet formulated."
        self.conversation = Conversation(
            time_enabled=True, autosave=False, save_enabled=False
        )

        # Enhanced cost model based on the paper's methodology
        self.test_cost_db = {
            "default": 150,
            "cbc": 50,
            "complete blood count": 50,
            "fbc": 50,
            "chest x-ray": 200,
            "chest xray": 200,
            "mri": 1500,
            "mri brain": 1800,
            "mri neck": 1600,
            "ct scan": 1200,
            "ct chest": 1300,
            "ct abdomen": 1400,
            "biopsy": 800,
            "core biopsy": 900,
            "immunohistochemistry": 400,
            "fish test": 500,
            "fish": 500,
            "ultrasound": 300,
            "ecg": 100,
            "ekg": 100,
            "blood glucose": 30,
            "liver function tests": 80,
            "renal function": 70,
            "toxic alcohol panel": 200,
            "urinalysis": 40,
            "culture": 150,
            "pathology": 600,
        }

        self._init_agents()
        logger.info(
            f"🏥 MAI Diagnostic Orchestrator initialized successfully in '{mode}' mode with budget ${initial_budget:,}"
        )

    def _init_agents(self):
        """Initializes all required agents with their specific roles and prompts."""
        self.agents = {
            role: Agent(
                agent_name=role.value,
                system_prompt=self._get_prompt_for_role(role),
                model_name=self.model_name,
                max_loops=1,
                output_type=(
                    "json" if role == AgentRole.CONSENSUS else "str"
                ),
                print_on=True,  # Enable printing for all agents to see outputs
            )
            for role in AgentRole
        }
        logger.info(
            f"👥 {len(self.agents)} virtual physician agents initialized and ready for consultation"
        )

    def _get_prompt_for_role(self, role: AgentRole) -> str:
        """Returns the system prompt for a given agent role."""
        prompts = {
            AgentRole.HYPOTHESIS: (
                """
            You are Dr. Hypothesis, a specialist in maintaining differential diagnoses. Your role is critical to the diagnostic process.

            CORE RESPONSIBILITIES:
            - Maintain a probability-ranked differential diagnosis with the top 3 most likely conditions
            - Update probabilities using Bayesian reasoning after each new finding
            - Consider both common and rare diseases appropriate to the clinical context
            - Explicitly track how new evidence changes your diagnostic thinking

            APPROACH:
            1. Start with the most likely diagnoses based on presenting symptoms
            2. For each new piece of evidence, consider:
               - How it supports or refutes each hypothesis
               - Whether it suggests new diagnoses to consider
               - How it changes the relative probabilities
            3. Always explain your Bayesian reasoning clearly

            OUTPUT FORMAT:
            Provide your updated differential diagnosis with:
            - Top 3 diagnoses with probability estimates (percentages)
            - Brief rationale for each
            - Key evidence supporting each hypothesis
            - Evidence that contradicts or challenges each hypothesis

            Remember: Your differential drives the entire diagnostic process. Be thorough, evidence-based, and adaptive.
            """
            ),
            AgentRole.TEST_CHOOSER: (
                """
            You are Dr. Test-Chooser, a specialist in diagnostic test selection and information theory.

            CORE RESPONSIBILITIES:
            - Select up to 3 diagnostic tests per round that maximally discriminate between leading hypotheses
            - Optimize for information value, not just clinical reasonableness
            - Consider test characteristics: sensitivity, specificity, positive/negative predictive values
            - Balance diagnostic yield with patient burden and resource utilization

            SELECTION CRITERIA:
            1. Information Value: How much will this test change diagnostic probabilities?
            2. Discriminatory Power: How well does it distinguish between competing hypotheses?
            3. Clinical Impact: Will the result meaningfully alter management?
            4. Sequential Logic: What should we establish first before ordering more complex tests?

            APPROACH:
            - For each proposed test, explicitly state which hypotheses it will help confirm or exclude
            - Consider both positive and negative results and their implications
            - Think about test sequences (e.g., basic labs before advanced imaging)
            - Avoid redundant tests that won't add new information

            OUTPUT FORMAT:
            For each recommended test:
            - Test name (be specific)
            - Primary hypotheses it will help evaluate
            - Expected information gain
            - How results will change management decisions

            Focus on tests that will most efficiently narrow the differential diagnosis.
            """
            ),
            AgentRole.CHALLENGER: (
                """
            You are Dr. Challenger, the critical thinking specialist and devil's advocate.

            CORE RESPONSIBILITIES:
            - Identify and challenge cognitive biases in the diagnostic process
            - Highlight contradictory evidence that might be overlooked
            - Propose alternative hypotheses and falsifying tests
            - Guard against premature diagnostic closure

            COGNITIVE BIASES TO WATCH FOR:
            1. Anchoring: Over-reliance on initial impressions
            2. Confirmation bias: Seeking only supporting evidence
            3. Availability bias: Overestimating probability of recently seen conditions
            4. Representativeness: Ignoring base rates and prevalence
            5. Search satisficing: Stopping at "good enough" explanations

            YOUR APPROACH:
            - Ask "What else could this be?" and "What doesn't fit?"
            - Challenge assumptions and look for alternative explanations
            - Propose tests that could disprove the leading hypothesis
            - Consider rare diseases when common ones don't fully explain the picture
            - Advocate for considering multiple conditions simultaneously

            OUTPUT FORMAT:
            - Specific biases you've identified in the current reasoning
            - Evidence that contradicts the leading hypotheses
            - Alternative diagnoses to consider
            - Tests that could falsify current assumptions
            - Red flags or concerning patterns that need attention

            Be constructively critical - your role is to strengthen diagnostic accuracy through rigorous challenge.
            """
            ),
            AgentRole.STEWARDSHIP: (
                """
            You are Dr. Stewardship, the resource optimization and cost-effectiveness specialist.

            CORE RESPONSIBILITIES:
            - Enforce cost-conscious, high-value care
            - Advocate for cheaper alternatives when diagnostically equivalent
            - Challenge low-yield, expensive tests
            - Balance diagnostic thoroughness with resource stewardship

            COST-VALUE FRAMEWORK:
            1. High-Value Tests: Low cost, high diagnostic yield, changes management
            2. Moderate-Value Tests: Moderate cost, specific indication, incremental value
            3. Low-Value Tests: High cost, low yield, minimal impact on decisions
            4. No-Value Tests: Any cost, no diagnostic value, ordered out of habit

            ALTERNATIVE STRATEGIES:
            - Could patient history/physical exam provide this information?
            - Is there a less expensive test with similar diagnostic value?
            - Can we use a staged approach (cheap test first, expensive if needed)?
            - Does the test result actually change management?

            YOUR APPROACH:
            - Review all proposed tests for necessity and value
            - Suggest cost-effective alternatives
            - Question tests that don't clearly advance diagnosis
            - Advocate for asking questions before ordering expensive tests
            - Consider the cumulative cost burden

            OUTPUT FORMAT:
            - Assessment of proposed tests (high/moderate/low/no value)
            - Specific cost-effective alternatives
            - Questions that might obviate need for testing
            - Recommended modifications to testing strategy
            - Cumulative cost considerations

            Your goal: Maximum diagnostic accuracy at minimum necessary cost.
            """
            ),
            AgentRole.CHECKLIST: (
                """
            You are Dr. Checklist, the quality assurance and consistency specialist.

            CORE RESPONSIBILITIES:
            - Perform silent quality control on all panel deliberations
            - Ensure test names are valid and properly specified
            - Check internal consistency of reasoning across panel members
            - Flag logical errors or contradictions in the diagnostic approach

            QUALITY CHECKS:
            1. Test Validity: Are proposed tests real and properly named?
            2. Logical Consistency: Do the recommendations align with the differential?
            3. Evidence Integration: Are all findings being considered appropriately?
            4. Process Adherence: Is the panel following proper diagnostic methodology?
            5. Safety Checks: Are any critical possibilities being overlooked?

            SPECIFIC VALIDATIONS:
            - Test names match standard medical terminology
            - Proposed tests are appropriate for the clinical scenario
            - No contradictions between different panel members' reasoning
            - All significant findings are being addressed
            - No gaps in the diagnostic logic

            OUTPUT FORMAT:
            - Brief validation summary (✓ Clear / ⚠ Issues noted)
            - Any test name corrections needed
            - Logical inconsistencies identified
            - Missing considerations or gaps
            - Process improvement suggestions

            Keep your feedback concise but comprehensive. Flag any issues that could compromise diagnostic quality.
            """
            ),
            AgentRole.CONSENSUS: (
                """
            You are the Consensus Coordinator, responsible for synthesizing the virtual panel's expertise into a single, optimal decision.

            CORE RESPONSIBILITIES:
            - Integrate input from Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, and Dr. Checklist
            - Decide on the single best next action: 'ask', 'test', or 'diagnose'
            - Balance competing priorities: accuracy, cost, efficiency, and thoroughness
            - Ensure the chosen action advances the diagnostic process optimally

            DECISION FRAMEWORK:
            1. DIAGNOSE: Choose when diagnostic certainty is sufficiently high (>85%) for the leading hypothesis
            2. TEST: Choose when tests will meaningfully discriminate between hypotheses
            3. ASK: Choose when history/exam questions could provide high-value information

            SYNTHESIS PROCESS:
            - Weight Dr. Hypothesis's confidence level and differential
            - Consider Dr. Test-Chooser's information value analysis
            - Incorporate Dr. Challenger's alternative perspectives
            - Respect Dr. Stewardship's cost-effectiveness concerns
            - Address any quality issues raised by Dr. Checklist

            OUTPUT REQUIREMENTS:
            Provide a JSON object with this exact structure:
            {
                "action_type": "ask" | "test" | "diagnose",
                "content": "specific question(s), test name(s), or final diagnosis",
                "reasoning": "clear justification synthesizing panel input"
            }

            For action_type "ask": content should be specific patient history or physical exam questions
            For action_type "test": content should be properly named diagnostic tests (up to 3)
            For action_type "diagnose": content should be the complete, specific final diagnosis

            Make the decision that best advances accurate, cost-effective diagnosis.
            """
            ),
            AgentRole.GATEKEEPER: (
                """
            You are the Gatekeeper, the clinical information oracle with complete access to the patient case file.

            CORE RESPONSIBILITIES:
            - Provide objective, specific clinical findings when explicitly requested
            - Serve as the authoritative source for all patient information
            - Generate realistic synthetic findings for tests not in the original case
            - Maintain clinical realism while preventing information leakage

            RESPONSE PRINCIPLES:
            1. OBJECTIVITY: Provide only factual findings, never interpretations or impressions
            2. SPECIFICITY: Give precise, detailed results when tests are properly ordered
            3. REALISM: Ensure all responses reflect realistic clinical scenarios
            4. NO HINTS: Never provide diagnostic clues or suggestions
            5. CONSISTENCY: Maintain coherence across all provided information

            HANDLING REQUESTS:
            - Patient History Questions: Provide relevant history from case file or realistic details
            - Physical Exam: Give specific examination findings as would be documented
            - Diagnostic Tests: Provide exact results as specified or realistic synthetic values
            - Vague Requests: Politely ask for more specific queries
            - Invalid Requests: Explain why the request cannot be fulfilled

            SYNTHETIC FINDINGS GUIDELINES:
            When generating findings not in the original case:
            - Ensure consistency with established diagnosis and case details
            - Use realistic reference ranges and values
            - Maintain clinical plausibility
            - Avoid pathognomonic findings unless specifically diagnostic

            RESPONSE FORMAT:
            - Direct, clinical language
            - Specific measurements with reference ranges when applicable
            - Clear organization of findings
            - Professional medical terminology

            Your role is crucial: provide complete, accurate clinical information while maintaining the challenge of the diagnostic process.
            """
            ),
            AgentRole.JUDGE: (
                """
            You are the Judge, the diagnostic accuracy evaluation specialist.

            CORE RESPONSIBILITIES:
            - Evaluate candidate diagnoses against ground truth using a rigorous clinical rubric
            - Provide fair, consistent scoring based on clinical management implications
            - Consider diagnostic substance over terminology differences
            - Account for acceptable medical synonyms and equivalent formulations

            EVALUATION RUBRIC (5-point Likert scale):

            SCORE 5 (Perfect/Clinically Superior):
            - Clinically identical to reference diagnosis
            - May be more specific than reference (adding relevant detail)
            - No incorrect or unrelated additions
            - Treatment approach would be identical

            SCORE 4 (Mostly Correct - Minor Incompleteness):
            - Core disease correctly identified
            - Minor qualifier or component missing/mis-specified
            - Overall management largely unchanged
            - Clinically appropriate diagnosis

            SCORE 3 (Partially Correct - Major Error):
            - Correct general disease category
            - Major error in etiology, anatomic site, or critical specificity
            - Would significantly alter workup or prognosis
            - Partially correct but clinically concerning gaps

            SCORE 2 (Largely Incorrect):
            - Shares only superficial features with correct diagnosis
            - Wrong fundamental disease process
            - Would misdirect clinical workup
            - Partially contradicts case details

            SCORE 1 (Completely Incorrect):
            - No meaningful overlap with correct diagnosis
            - Wrong organ system or disease category
            - Would likely lead to harmful care
            - Completely inconsistent with clinical presentation

            EVALUATION PROCESS:
            1. Compare core disease entity
            2. Assess etiology/causative factors
            3. Evaluate anatomic specificity
            4. Consider diagnostic completeness
            5. Judge clinical management implications

            OUTPUT FORMAT:
            - Score (1-5) with clear label
            - Detailed justification referencing specific rubric criteria
            - Explanation of how diagnosis would affect clinical management
            - Note any acceptable medical synonyms or equivalent terminology

            Maintain high standards while recognizing legitimate diagnostic variability in medical practice.
            """
            ),
        }
        return prompts[role]

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Safely parses a JSON string, returning a dictionary."""
        try:
            # Extract the actual response content from the agent response
            if isinstance(response, str):
                # Handle markdown-formatted JSON
                if "```json" in response:
                    # Extract JSON content between ```json and ```
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response.find(start_marker)
                    if start_idx != -1:
                        start_idx += len(start_marker)
                        end_idx = response.find(end_marker, start_idx)
                        if end_idx != -1:
                            json_content = response[
                                start_idx:end_idx
                            ].strip()
                            return json.loads(json_content)

                # Try to find JSON-like content in the response
                lines = response.split("\n")
                json_lines = []
                in_json = False
                brace_count = 0

                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("{") and not in_json:
                        in_json = True
                        json_lines = [line]  # Start fresh
                        brace_count = line.count("{") - line.count(
                            "}"
                        )
                    elif in_json:
                        json_lines.append(line)
                        brace_count += line.count("{") - line.count(
                            "}"
                        )
                        if (
                            brace_count <= 0
                        ):  # Balanced braces, end of JSON
                            break

                if json_lines and in_json:
                    json_content = "\n".join(json_lines)
                    return json.loads(json_content)

                # Try to extract JSON from text that might contain other content
                import re

                # Look for JSON pattern in the text
                json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
                matches = re.findall(
                    json_pattern, response, re.DOTALL
                )

                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue

                # Direct parsing attempt as fallback
                return json.loads(response)

        except (
            json.JSONDecodeError,
            IndexError,
            AttributeError,
        ) as e:
            logger.error(f"Failed to parse JSON response. Error: {e}")
            logger.debug(
                f"Response content: {response[:500]}..."
            )  # Log first 500 chars
            # Fallback to a default action if parsing fails
            return {
                "action_type": "ask",
                "content": (
                    "Could you please clarify the next best step? The previous analysis was inconclusive."
                ),
                "reasoning": "Fallback due to parsing error.",
            }

    def _estimate_cost(self, tests: Union[List[str], str]) -> int:
        """Estimates the cost of diagnostic tests."""
        if isinstance(tests, str):
            tests = [tests]

        cost = 0
        for test in tests:
            test_lower = test.lower().strip()

            # Enhanced cost matching with multiple strategies
            cost_found = False

            # Strategy 1: Exact match
            if test_lower in self.test_cost_db:
                cost += self.test_cost_db[test_lower]
                cost_found = True
                continue

            # Strategy 2: Partial match (find best matching key)
            best_match = None
            best_match_length = 0
            for cost_key in self.test_cost_db:
                if cost_key in test_lower or test_lower in cost_key:
                    if len(cost_key) > best_match_length:
                        best_match = cost_key
                        best_match_length = len(cost_key)

            if best_match:
                cost += self.test_cost_db[best_match]
                cost_found = True
                continue

            # Strategy 3: Keyword-based matching
            if any(
                keyword in test_lower
                for keyword in ["biopsy", "tissue"]
            ):
                cost += self.test_cost_db.get("biopsy", 800)
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["mri", "magnetic"]
            ):
                cost += self.test_cost_db.get("mri", 1500)
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["ct", "computed tomography"]
            ):
                cost += self.test_cost_db.get("ct scan", 1200)
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["xray", "x-ray", "radiograph"]
            ):
                cost += self.test_cost_db.get("chest x-ray", 200)
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["blood", "serum", "plasma"]
            ):
                cost += 100  # Basic blood test cost
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["culture", "sensitivity"]
            ):
                cost += self.test_cost_db.get("culture", 150)
                cost_found = True
            elif any(
                keyword in test_lower
                for keyword in ["immunohistochemistry", "ihc"]
            ):
                cost += self.test_cost_db.get(
                    "immunohistochemistry", 400
                )
                cost_found = True

            # Strategy 4: Default cost for unknown tests
            if not cost_found:
                cost += self.test_cost_db["default"]
                logger.debug(
                    f"Using default cost for unknown test: {test}"
                )

        return cost

    def _run_panel_deliberation(self) -> Action:
        """Orchestrates one round of debate among the virtual panel to decide the next action."""
        logger.info(
            "🩺 Virtual medical panel deliberation commenced - analyzing patient case"
        )
        logger.debug(
            "Panel members: Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, Dr. Checklist"
        )
        panel_conversation = Conversation(
            time_enabled=True, autosave=False, save_enabled=False
        )

        # Prepare comprehensive panel context
        remaining_budget = self.initial_budget - self.cumulative_cost
        budget_status = (
            "EXCEEDED"
            if remaining_budget < 0
            else f"${remaining_budget:,}"
        )

        panel_context = f"""
        DIAGNOSTIC CASE STATUS - ROUND {len(self.conversation.return_history_as_string().split('Gatekeeper:')) - 1}
        
        === CASE INFORMATION ===
        {self.conversation.get_str()}
        
        === CURRENT STATE ===
        Differential Diagnosis: {self.differential_diagnosis}
        Cumulative Cost: ${self.cumulative_cost:,}
        Remaining Budget: {budget_status}
        Mode: {self.mode}
        Max Iterations: {self.max_iterations}
        
        === PANEL TASK ===
        Virtual medical panel, please deliberate systematically on the next best diagnostic action.
        Each specialist should provide their expert analysis in sequence.
        """
        panel_conversation.add("System", panel_context)

        # Check mode-specific constraints
        if self.mode == "instant":
            # For instant mode, skip deliberation and go straight to diagnosis
            action_dict = {
                "action_type": "diagnose",
                "content": (
                    self.differential_diagnosis.split("\n")[0]
                    if "\n" in self.differential_diagnosis
                    else self.differential_diagnosis
                ),
                "reasoning": (
                    "Instant diagnosis mode - providing immediate assessment based on initial presentation"
                ),
            }
            return Action(**action_dict)

        if self.mode == "question_only":
            # For question-only mode, prevent test ordering
            panel_context += "\n\nIMPORTANT: This is QUESTION-ONLY mode. You may ONLY ask patient questions, not order diagnostic tests."
            panel_conversation.add("System", panel_context)

        # Sequential expert deliberation with enhanced methodology
        try:
            # Dr. Hypothesis - Differential diagnosis and probability assessment
            logger.info(
                "🧠 Dr. Hypothesis analyzing differential diagnosis..."
            )
            hypothesis = self.agents[AgentRole.HYPOTHESIS].run(
                panel_conversation.get_str()
            )
            self.differential_diagnosis = (
                hypothesis  # Update main state
            )
            panel_conversation.add(
                self.agents[AgentRole.HYPOTHESIS].agent_name,
                hypothesis,
            )

            # Dr. Test-Chooser - Information value optimization
            logger.info(
                "🔬 Dr. Test-Chooser selecting optimal tests..."
            )
            test_choices = self.agents[AgentRole.TEST_CHOOSER].run(
                panel_conversation.get_str()
            )
            panel_conversation.add(
                self.agents[AgentRole.TEST_CHOOSER].agent_name,
                test_choices,
            )

            # Dr. Challenger - Bias identification and alternative hypotheses
            logger.info(
                "🤔 Dr. Challenger challenging assumptions..."
            )
            challenges = self.agents[AgentRole.CHALLENGER].run(
                panel_conversation.get_str()
            )
            panel_conversation.add(
                self.agents[AgentRole.CHALLENGER].agent_name,
                challenges,
            )

            # Dr. Stewardship - Cost-effectiveness analysis
            logger.info(
                "💰 Dr. Stewardship evaluating cost-effectiveness..."
            )
            stewardship_context = panel_conversation.get_str()
            if self.enable_budget_tracking:
                stewardship_context += f"\n\nBUDGET TRACKING ENABLED - Current cost: ${self.cumulative_cost}, Remaining: ${remaining_budget}"
            stewardship_rec = self.agents[AgentRole.STEWARDSHIP].run(
                stewardship_context
            )
            panel_conversation.add(
                self.agents[AgentRole.STEWARDSHIP].agent_name,
                stewardship_rec,
            )

            # Dr. Checklist - Quality assurance
            logger.info(
                "✅ Dr. Checklist performing quality control..."
            )
            checklist_rep = self.agents[AgentRole.CHECKLIST].run(
                panel_conversation.get_str()
            )
            panel_conversation.add(
                self.agents[AgentRole.CHECKLIST].agent_name,
                checklist_rep,
            )

            # Consensus Coordinator - Final decision synthesis
            logger.info(
                "🤝 Consensus Coordinator synthesizing panel decision..."
            )
            consensus_context = panel_conversation.get_str()

            # Add mode-specific constraints to consensus
            if self.mode == "budgeted" and remaining_budget <= 0:
                consensus_context += "\n\nBUDGET CONSTRAINT: Budget exceeded - must either ask questions or provide final diagnosis."

            consensus_response = self.agents[AgentRole.CONSENSUS].run(
                consensus_context
            )
            logger.debug(
                f"Raw consensus response: {consensus_response}"
            )

            # Extract the actual text content from agent response
            if hasattr(consensus_response, "content"):
                response_text = consensus_response.content
            elif isinstance(consensus_response, str):
                response_text = consensus_response
            else:
                response_text = str(consensus_response)

            action_dict = self._parse_json_response(response_text)

            # Validate action based on mode constraints
            action = Action(**action_dict)
            if (
                self.mode == "question_only"
                and action.action_type == "test"
            ):
                logger.warning(
                    "Test ordering attempted in question-only mode, converting to ask action"
                )
                action.action_type = "ask"
                action.content = "Can you provide more details about the patient's symptoms and history?"
                action.reasoning = (
                    "Mode constraint: question-only mode active"
                )

            if (
                self.mode == "budgeted"
                and action.action_type == "test"
                and remaining_budget <= 0
            ):
                logger.warning(
                    "Test ordering attempted with insufficient budget, converting to diagnose action"
                )
                action.action_type = "diagnose"
                action.content = (
                    self.differential_diagnosis.split("\n")[0]
                    if "\n" in self.differential_diagnosis
                    else self.differential_diagnosis
                )
                action.reasoning = "Budget constraint: insufficient funds for additional testing"

            return action

        except Exception as e:
            logger.error(f"Error during panel deliberation: {e}")
            # Fallback action
            return Action(
                action_type="ask",
                content="Could you please provide more information about the patient's current condition?",
                reasoning=f"Fallback due to panel deliberation error: {str(e)}",
            )

    def _interact_with_gatekeeper(
        self, action: Action, full_case_details: str
    ) -> str:
        """Sends the panel's action to the Gatekeeper and returns its response."""
        gatekeeper = self.agents[AgentRole.GATEKEEPER]

        if action.action_type == "ask":
            request = f"Question: {action.content}"
        elif action.action_type == "test":
            request = f"Tests ordered: {', '.join(action.content)}"
        else:
            return "No interaction needed for 'diagnose' action."

        # The Gatekeeper needs the full case to act as an oracle
        prompt = f"""
        Full Case Details (for your reference only):
        ---
        {full_case_details}
        ---
        
        Request from Diagnostic Agent:
        {request}
        """

        response = gatekeeper.run(prompt)
        return response

    def _judge_diagnosis(
        self, candidate_diagnosis: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Uses the Judge agent to evaluate the final diagnosis."""
        judge = self.agents[AgentRole.JUDGE]
        prompt = f"""
        Please evaluate the following diagnosis.
        Ground Truth: "{ground_truth}"
        Candidate Diagnosis: "{candidate_diagnosis}"
        """
        response = judge.run(prompt)

        # Simple parsing for demonstration; a more robust solution would use structured output.
        try:
            score = float(
                response.split("Score:")[1].split("/")[0].strip()
            )
            reasoning = response.split("Justification:")[1].strip()
        except (IndexError, ValueError):
            score = 0.0
            reasoning = "Could not parse judge's response."

        return {"score": score, "reasoning": reasoning}

    def run(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
    ) -> DiagnosisResult:
        """
        Executes the full sequential diagnostic process.

        Args:
            initial_case_info (str): The initial abstract of the case.
            full_case_details (str): The complete case file for the Gatekeeper.
            ground_truth_diagnosis (str): The correct final diagnosis for evaluation.

        Returns:
            DiagnosisResult: An object containing the final diagnosis, evaluation, cost, and history.
        """
        start_time = time.time()
        self.conversation.add(
            "Gatekeeper",
            f"Initial Case Information: {initial_case_info}",
        )

        # Add initial physician visit cost
        self.cumulative_cost += self.physician_visit_cost
        logger.info(
            f"Initial physician visit cost: ${self.physician_visit_cost}"
        )

        final_diagnosis = None
        iteration_count = 0

        for i in range(self.max_iterations):
            iteration_count = i + 1
            logger.info(
                f"--- Starting Diagnostic Loop {iteration_count}/{self.max_iterations} ---"
            )
            logger.info(
                f"Current cost: ${self.cumulative_cost:,} | Remaining budget: ${self.initial_budget - self.cumulative_cost:,}"
            )

            try:
                # Panel deliberates to decide on the next action
                action = self._run_panel_deliberation()
                logger.info(
                    f"⚕️ Panel decision: {action.action_type.upper()} -> {action.content}"
                )
                logger.info(
                    f"💭 Medical reasoning: {action.reasoning}"
                )

                if action.action_type == "diagnose":
                    final_diagnosis = action.content
                    logger.info(
                        f"Final diagnosis proposed: {final_diagnosis}"
                    )
                    break

                # Handle mode-specific constraints
                if (
                    self.mode == "question_only"
                    and action.action_type == "test"
                ):
                    logger.warning(
                        "Test ordering blocked in question-only mode"
                    )
                    continue

                if (
                    self.mode == "budgeted"
                    and action.action_type == "test"
                ):
                    # Check if we can afford the tests
                    estimated_test_cost = self._estimate_cost(
                        action.content
                    )
                    if (
                        self.cumulative_cost + estimated_test_cost
                        > self.initial_budget
                    ):
                        logger.warning(
                            f"Test cost ${estimated_test_cost} would exceed budget. Skipping tests."
                        )
                        continue

                # Interact with the Gatekeeper
                response = self._interact_with_gatekeeper(
                    action, full_case_details
                )
                self.conversation.add("Gatekeeper", response)

                # Update costs based on action type
                if action.action_type == "test":
                    test_cost = self._estimate_cost(action.content)
                    self.cumulative_cost += test_cost
                    logger.info(f"Tests ordered: {action.content}")
                    logger.info(
                        f"Test cost: ${test_cost:,} | Cumulative cost: ${self.cumulative_cost:,}"
                    )
                elif action.action_type == "ask":
                    # Questions are part of the same visit until tests are ordered
                    logger.info(f"Questions asked: {action.content}")
                    logger.info(
                        "No additional cost for questions in same visit"
                    )

                # Check budget constraints for budgeted mode
                if (
                    self.mode == "budgeted"
                    and self.cumulative_cost >= self.initial_budget
                ):
                    logger.warning(
                        "Budget limit reached. Forcing final diagnosis."
                    )
                    # Use current differential diagnosis or make best guess
                    final_diagnosis = (
                        self.differential_diagnosis.split("\n")[0]
                        if "\n" in self.differential_diagnosis
                        else "Diagnosis not reached within budget constraints."
                    )
                    break

            except Exception as e:
                logger.error(
                    f"Error in diagnostic loop {iteration_count}: {e}"
                )
                # Continue to next iteration or break if critical error
                continue

        else:
            # Max iterations reached without diagnosis
            final_diagnosis = (
                self.differential_diagnosis.split("\n")[0]
                if "\n" in self.differential_diagnosis
                else "Diagnosis not reached within maximum iterations."
            )
            logger.warning(
                f"Max iterations ({self.max_iterations}) reached. Using best available diagnosis."
            )

        # Ensure we have a final diagnosis
        if not final_diagnosis or final_diagnosis.strip() == "":
            final_diagnosis = (
                "Unable to determine diagnosis within constraints."
            )

        # Calculate total time
        total_time = time.time() - start_time
        logger.info(
            f"Diagnostic session completed in {total_time:.2f} seconds"
        )

        # Judge the final diagnosis
        logger.info("Evaluating final diagnosis...")
        try:
            judgement = self._judge_diagnosis(
                final_diagnosis, ground_truth_diagnosis
            )
        except Exception as e:
            logger.error(f"Error in diagnosis evaluation: {e}")
            judgement = {
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
            }

        # Create comprehensive result
        result = DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth_diagnosis,
            accuracy_score=judgement["score"],
            accuracy_reasoning=judgement["reasoning"],
            total_cost=self.cumulative_cost,
            iterations=iteration_count,
            conversation_history=self.conversation.get_str(),
        )

        logger.info("Diagnostic process completed:")
        logger.info(f"  Final diagnosis: {final_diagnosis}")
        logger.info(f"  Ground truth: {ground_truth_diagnosis}")
        logger.info(f"  Accuracy score: {judgement['score']}/5.0")
        logger.info(f"  Total cost: ${self.cumulative_cost:,}")
        logger.info(f"  Iterations: {iteration_count}")

        return result

    def run_ensemble(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
        num_runs: int = 3,
    ) -> DiagnosisResult:
        """
        Runs multiple independent diagnostic sessions and aggregates the results.

        Args:
            initial_case_info (str): The initial abstract of the case.
            full_case_details (str): The complete case file for the Gatekeeper.
            ground_truth_diagnosis (str): The correct final diagnosis for evaluation.
            num_runs (int): Number of independent runs to perform.

        Returns:
            DiagnosisResult: Aggregated result from ensemble runs.
        """
        logger.info(
            f"Starting ensemble run with {num_runs} independent sessions"
        )

        ensemble_results = []
        total_cost = 0

        for run_id in range(num_runs):
            logger.info(
                f"=== Ensemble Run {run_id + 1}/{num_runs} ==="
            )

            # Create a fresh orchestrator instance for each run
            run_orchestrator = MaiDxOrchestrator(
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                initial_budget=self.initial_budget,
                mode="no_budget",  # Use no_budget for ensemble runs
                physician_visit_cost=self.physician_visit_cost,
                enable_budget_tracking=False,
            )

            # Run the diagnostic session
            result = run_orchestrator.run(
                initial_case_info,
                full_case_details,
                ground_truth_diagnosis,
            )
            ensemble_results.append(result)
            total_cost += result.total_cost

            logger.info(
                f"Run {run_id + 1} completed: {result.final_diagnosis} (Score: {result.accuracy_score})"
            )

        # Aggregate results using consensus
        final_diagnosis = self._aggregate_ensemble_diagnoses(
            [r.final_diagnosis for r in ensemble_results]
        )

        # Judge the aggregated diagnosis
        judgement = self._judge_diagnosis(
            final_diagnosis, ground_truth_diagnosis
        )

        # Calculate average metrics
        avg_iterations = sum(
            r.iterations for r in ensemble_results
        ) / len(ensemble_results)

        # Combine conversation histories
        combined_history = "\n\n=== ENSEMBLE RESULTS ===\n"
        for i, result in enumerate(ensemble_results):
            combined_history += f"\n--- Run {i+1} ---\n"
            combined_history += (
                f"Diagnosis: {result.final_diagnosis}\n"
            )
            combined_history += f"Score: {result.accuracy_score}\n"
            combined_history += f"Cost: ${result.total_cost:,}\n"
            combined_history += f"Iterations: {result.iterations}\n"

        combined_history += "\n--- Aggregated Result ---\n"
        combined_history += f"Final Diagnosis: {final_diagnosis}\n"
        combined_history += f"Reasoning: {judgement['reasoning']}\n"

        ensemble_result = DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth_diagnosis,
            accuracy_score=judgement["score"],
            accuracy_reasoning=judgement["reasoning"],
            total_cost=total_cost,  # Sum of all runs
            iterations=int(avg_iterations),
            conversation_history=combined_history,
        )

        logger.info(
            f"Ensemble completed: {final_diagnosis} (Score: {judgement['score']})"
        )
        return ensemble_result

    def _aggregate_ensemble_diagnoses(
        self, diagnoses: List[str]
    ) -> str:
        """Aggregates multiple diagnoses from ensemble runs."""
        # Simple majority voting or use the most confident diagnosis
        if not diagnoses:
            return "No diagnosis available"

        # Remove any empty or invalid diagnoses
        valid_diagnoses = [
            d
            for d in diagnoses
            if d and d.strip() and "not reached" not in d.lower()
        ]

        if not valid_diagnoses:
            return diagnoses[0] if diagnoses else "No valid diagnosis"

        # If all diagnoses are the same, return that
        if len(set(valid_diagnoses)) == 1:
            return valid_diagnoses[0]

        # Use an aggregator agent to select the best diagnosis
        try:
            aggregator_prompt = f"""
            You are a medical consensus aggregator. Given multiple diagnostic assessments from independent medical panels, 
            select the most accurate and complete diagnosis.
            
            Diagnoses to consider:
            {chr(10).join(f"{i+1}. {d}" for i, d in enumerate(valid_diagnoses))}
            
            Provide the single best diagnosis that represents the medical consensus. 
            Consider clinical accuracy, specificity, and completeness.
            """

            aggregator = Agent(
                agent_name="Ensemble Aggregator",
                system_prompt=aggregator_prompt,
                model_name=self.model_name,
                max_loops=1,
                print_on=True,  # Enable printing for aggregator agent
            )

            return aggregator.run(aggregator_prompt).strip()

        except Exception as e:
            logger.error(f"Error in ensemble aggregation: {e}")
            # Fallback to most common diagnosis
            from collections import Counter

            return Counter(valid_diagnoses).most_common(1)[0][0]

    @classmethod
    def create_variant(
        cls, variant: str, **kwargs
    ) -> "MaiDxOrchestrator":
        """
        Factory method to create different MAI-DxO variants as described in the paper.

        Args:
            variant (str): One of 'instant', 'question_only', 'budgeted', 'no_budget', 'ensemble'
            **kwargs: Additional parameters for the orchestrator

        Returns:
            MaiDxOrchestrator: Configured orchestrator instance
        """
        variant_configs = {
            "instant": {
                "mode": "instant",
                "max_iterations": 1,
                "enable_budget_tracking": False,
            },
            "question_only": {
                "mode": "question_only",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
            "budgeted": {
                "mode": "budgeted",
                "max_iterations": 10,
                "enable_budget_tracking": True,
                "initial_budget": kwargs.get("budget", 5000),
            },
            "no_budget": {
                "mode": "no_budget",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
            "ensemble": {
                "mode": "no_budget",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
        }

        if variant not in variant_configs:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from: {list(variant_configs.keys())}"
            )

        config = variant_configs[variant]
        config.update(kwargs)  # Allow overrides

        return cls(**config)


def run_mai_dxo_demo(
    case_info: str = None,
    case_details: str = None,
    ground_truth: str = None,
) -> Dict[str, DiagnosisResult]:
    """
    Convenience function to run a quick demonstration of MAI-DxO variants.

    Args:
        case_info (str): Initial case information. Uses default if None.
        case_details (str): Full case details. Uses default if None.
        ground_truth (str): Ground truth diagnosis. Uses default if None.

    Returns:
        Dict[str, DiagnosisResult]: Results from different MAI-DxO variants
    """
    # Use default case if not provided
    if not case_info:
        case_info = (
            "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
            "and bleeding. Symptoms did not abate with antimicrobial therapy."
        )

    if not case_details:
        case_details = """
        Patient: 29-year-old female.
        History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
        No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable.
        Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
        Initial Labs: FBC, clotting studies normal.
        MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
        Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
        Biopsy (Immunohistochemistry): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
        Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
        Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
        """

    if not ground_truth:
        ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    results = {}

    # Test key variants
    variants = ["no_budget", "budgeted", "question_only"]

    for variant in variants:
        try:
            logger.info(f"Running MAI-DxO variant: {variant}")

            if variant == "budgeted":
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant,
                    budget=3000,
                    model_name="gemini/gemini-2.5-flash",
                )
            else:
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant, model_name="gemini/gemini-2.5-flash"
                )

            result = orchestrator.run(
                case_info, case_details, ground_truth
            )
            results[variant] = result

        except Exception as e:
            logger.error(f"Error running variant {variant}: {e}")
            results[variant] = None

    return results


if __name__ == "__main__":
    # Example case inspired by the paper's Figure 1
    initial_info = (
        "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
        "and bleeding. Symptoms did not abate with antimicrobial therapy."
    )

    full_case = """
    Patient: 29-year-old female.
    History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
    No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable. No history of smoking or significant alcohol use.
    Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
    Initial Labs: FBC, clotting studies normal.
    MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
    Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
    Biopsy (Immunohistochemistry for Carcinoma): CD31, D2-40, CD34, ERG, GLUT-1, pan-cytokeratin, CD45, CD20, CD3 all negative. Ki-67: 60% nuclear positivity.
    Biopsy (Immunohistochemistry for Rhabdomyosarcoma): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
    Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
    Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
    """

    ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    # --- Demonstrate Different MAI-DxO Variants ---
    try:
        print("\n" + "=" * 80)
        print(
            "    MAI DIAGNOSTIC ORCHESTRATOR (MAI-DxO) - SEQUENTIAL DIAGNOSIS BENCHMARK"
        )
        print(
            "                    Implementation based on the NEJM Research Paper"
        )
        print("=" * 80)

        # Test different variants as described in the paper
        variants_to_test = [
            (
                "no_budget",
                "Standard MAI-DxO with no budget constraints",
            ),
            ("budgeted", "Budget-constrained MAI-DxO ($3000 limit)"),
            (
                "question_only",
                "Question-only variant (no diagnostic tests)",
            ),
        ]

        results = {}

        for variant_name, description in variants_to_test:
            print(f"\n{'='*60}")
            print(f"Testing Variant: {variant_name.upper()}")
            print(f"Description: {description}")
            print("=" * 60)

            # Create the variant
            if variant_name == "budgeted":
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant_name,
                    budget=3000,
                    model_name="gemini/gemini-2.5-flash",
                    max_iterations=5,
                )
            else:
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant_name,
                    model_name="gemini/gemini-2.5-flash",
                    max_iterations=5,
                )

            # Run the diagnostic process
            result = orchestrator.run(
                initial_case_info=initial_info,
                full_case_details=full_case,
                ground_truth_diagnosis=ground_truth,
            )

            results[variant_name] = result

            # Display results
            print(f"\n🚀 Final Diagnosis: {result.final_diagnosis}")
            print(f"🎯 Ground Truth: {result.ground_truth}")
            print(f"⭐ Accuracy Score: {result.accuracy_score}/5.0")
            print(f"   Reasoning: {result.accuracy_reasoning}")
            print(f"💰 Total Cost: ${result.total_cost:,}")
            print(f"🔄 Iterations: {result.iterations}")
            print(f"⏱️  Mode: {orchestrator.mode}")

        # Demonstrate ensemble approach
        print(f"\n{'='*60}")
        print("Testing Variant: ENSEMBLE")
        print(
            "Description: Multiple independent runs with consensus aggregation"
        )
        print("=" * 60)

        ensemble_orchestrator = MaiDxOrchestrator.create_variant(
            "ensemble",
            model_name="gemini/gemini-2.5-flash",
            max_iterations=3,  # Shorter iterations for ensemble
        )

        ensemble_result = ensemble_orchestrator.run_ensemble(
            initial_case_info=initial_info,
            full_case_details=full_case,
            ground_truth_diagnosis=ground_truth,
            num_runs=2,  # Reduced for demo
        )

        results["ensemble"] = ensemble_result

        print(
            f"\n🚀 Ensemble Diagnosis: {ensemble_result.final_diagnosis}"
        )
        print(f"🎯 Ground Truth: {ensemble_result.ground_truth}")
        print(
            f"⭐ Ensemble Score: {ensemble_result.accuracy_score}/5.0"
        )
        print(
            f"💰 Total Ensemble Cost: ${ensemble_result.total_cost:,}"
        )

        # --- Summary Comparison ---
        print(f"\n{'='*80}")
        print("                           RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Variant':<15} {'Diagnosis Match':<15} {'Score':<8} {'Cost':<12} {'Iterations':<12}"
        )
        print("-" * 80)

        for variant_name, result in results.items():
            match_status = (
                "✓ Match"
                if result.accuracy_score >= 4.0
                else "✗ No Match"
            )
            print(
                f"{variant_name:<15} {match_status:<15} {result.accuracy_score:<8.1f} ${result.total_cost:<11,} {result.iterations:<12}"
            )

        print(f"\n{'='*80}")
        print(
            "Implementation successfully demonstrates the MAI-DxO framework"
        )
        print(
            "as described in 'Sequential Diagnosis with Language Models' paper"
        )
        print("=" * 80)

    except Exception as e:
        logger.exception(
            f"An error occurred during the diagnostic session: {e}"
        )
        print(f"\n❌ Error occurred: {e}")
        print("Please check your model configuration and API keys.")
