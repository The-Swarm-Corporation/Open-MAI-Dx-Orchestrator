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
    orchestrator = MaiDxOrchestrator(model_name="gpt-4o")
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union, Literal

from loguru import logger
from pydantic import BaseModel, Field
from swarms import Agent, Conversation
from dotenv import load_dotenv

load_dotenv()

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
class CaseState:
    """Structured state management for diagnostic process - addresses Category 2.1"""
    initial_vignette: str
    evidence_log: List[str] = field(default_factory=list)
    differential_diagnosis: Dict[str, float] = field(default_factory=dict)
    tests_performed: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    cumulative_cost: int = 0
    iteration: int = 0
    last_actions: List['Action'] = field(default_factory=list)  # For stagnation detection
    
    def add_evidence(self, evidence: str):
        """Add new evidence to the case"""
        self.evidence_log.append(f"[Turn {self.iteration}] {evidence}")
    
    def update_differential(self, diagnosis_dict: Dict[str, float]):
        """Update differential diagnosis probabilities"""
        self.differential_diagnosis.update(diagnosis_dict)
    
    def add_test(self, test_name: str):
        """Record a test that was performed"""
        self.tests_performed.append(test_name)
    
    def add_question(self, question: str):
        """Record a question that was asked"""
        self.questions_asked.append(question)
        
    def is_stagnating(self, new_action: 'Action') -> bool:
        """Detect if the system is stuck in a loop - addresses Category 1.2"""
        if len(self.last_actions) < 2:
            return False
            
        # Check if the new action is identical to recent ones
        for last_action in self.last_actions[-2:]:
            if (last_action.action_type == new_action.action_type and 
                last_action.content == new_action.content):
                return True
        return False
    
    def add_action(self, action: 'Action'):
        """Add action to history and maintain sliding window"""
        self.last_actions.append(action)
        if len(self.last_actions) > 3:  # Keep only last 3 actions
            self.last_actions.pop(0)
    
    def get_max_confidence(self) -> float:
        """Get the maximum confidence from differential diagnosis"""
        if not self.differential_diagnosis:
            return 0.0
        return max(self.differential_diagnosis.values())
    
    def get_leading_diagnosis(self) -> str:
        """Get the diagnosis with highest confidence"""
        if not self.differential_diagnosis:
            return "No diagnosis formulated"
        return max(self.differential_diagnosis.items(), key=lambda x: x[1])[0]
    
    def summarize_evidence(self) -> str:
        """Create a concise summary of evidence for token efficiency"""
        if len(self.evidence_log) <= 5:
            return "\n".join(self.evidence_log)
        
        # Keep first 2 and last 3 entries, summarize middle
        summary_parts = []
        summary_parts.extend(self.evidence_log[:2])
        
        if len(self.evidence_log) > 5:
            middle_count = len(self.evidence_log) - 5
            summary_parts.append(f"[... {middle_count} additional findings ...]")
        
        summary_parts.extend(self.evidence_log[-3:])
        return "\n".join(summary_parts)


@dataclass
class DeliberationState:
    """Structured state for panel deliberation - addresses Category 1.1"""
    hypothesis_analysis: str = ""
    test_chooser_analysis: str = ""
    challenger_analysis: str = ""
    stewardship_analysis: str = ""
    checklist_analysis: str = ""
    situational_context: str = ""
    stagnation_detected: bool = False
    retry_count: int = 0
    
    def to_consensus_prompt(self) -> str:
        """Generate a structured prompt for the consensus coordinator - no truncation, let agent self-regulate"""
        
        prompt = f"""
You are the Consensus Coordinator. Here is the panel's analysis:

**Differential Diagnosis (Dr. Hypothesis):**
{self.hypothesis_analysis or 'Not yet formulated'}

**Test Recommendations (Dr. Test-Chooser):**
{self.test_chooser_analysis or 'None provided'}

**Critical Challenges (Dr. Challenger):**
{self.challenger_analysis or 'None identified'}

**Cost Assessment (Dr. Stewardship):**
{self.stewardship_analysis or 'Not evaluated'}

**Quality Control (Dr. Checklist):**
{self.checklist_analysis or 'No issues noted'}
"""
        
        if self.stagnation_detected:
            prompt += "\n**STAGNATION DETECTED** - The panel is repeating actions. You MUST make a decisive choice or provide final diagnosis."
        
        if self.situational_context:
            prompt += f"\n**Situational Context:** {self.situational_context}"
        
        prompt += "\n\nBased on this comprehensive panel input, use the make_consensus_decision function to provide your structured action."
        return prompt


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
    
    Enhanced with structured deliberation and proper state management as per research paper.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",  # Fixed: Use valid GPT-4 Turbo model name
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
        physician_visit_cost: int = 300,
        enable_budget_tracking: bool = False,
        request_delay: float = 8.0,  # seconds to wait between model calls to mitigate rate-limits
    ):
        """
        Initializes the MAI-DxO system with improved architecture.

        Args:
            model_name (str): The language model to be used by all agents.
            max_iterations (int): The maximum number of diagnostic loops.
            initial_budget (int): The starting budget for diagnostic tests.
            mode (str): The operational mode of MAI-DxO.
            physician_visit_cost (int): Cost per physician visit.
            enable_budget_tracking (bool): Whether to enable budget tracking.
            request_delay (float): Seconds to wait between model calls to mitigate rate-limits.
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.enable_budget_tracking = enable_budget_tracking

        # Throttle settings to avoid OpenAI TPM rate-limits
        self.request_delay = max(request_delay, 0)
        
        # Token management
        self.max_total_tokens_per_request = 25000  # Safety margin below 30k limit

        self.cumulative_cost = 0
        self.differential_diagnosis = "Not yet formulated."
        self.conversation = Conversation(
            time_enabled=True, autosave=False, save_enabled=False
        )
        
        # Initialize case state for structured state management
        self.case_state = None

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

    def _get_agent_max_tokens(self, role: AgentRole) -> int:
        """Get max_tokens for each agent based on their role - agents will self-regulate based on token guidance"""
        token_limits = {
            # Reasonable limits - agents will adjust their verbosity based on token guidance
            AgentRole.HYPOTHESIS: 1200,   # Function calling keeps this structured, but allow room for quality
            AgentRole.TEST_CHOOSER: 800,  # Need space for test rationale
            AgentRole.CHALLENGER: 800,    # Need space for critical analysis
            AgentRole.STEWARDSHIP: 600,
            AgentRole.CHECKLIST: 400,
            AgentRole.CONSENSUS: 500,     # Function calling is efficient
            AgentRole.GATEKEEPER: 1000,  # Needs to provide detailed clinical findings
            AgentRole.JUDGE: 700,
        }
        return token_limits.get(role, 600)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def _generate_token_guidance(self, input_tokens: int, max_output_tokens: int, total_tokens: int, agent_role: AgentRole) -> str:
        """Generate dynamic token guidance for agents to self-regulate their responses"""
        
        # Determine urgency level based on token usage
        if total_tokens > self.max_total_tokens_per_request:
            urgency = "CRITICAL"
            strategy = "Be extremely concise. Prioritize only the most essential information."
        elif total_tokens > self.max_total_tokens_per_request * 0.8:
            urgency = "HIGH"
            strategy = "Be concise and focus on key points. Avoid elaborate explanations."
        elif total_tokens > self.max_total_tokens_per_request * 0.6:
            urgency = "MODERATE"
            strategy = "Be reasonably concise while maintaining necessary detail."
        else:
            urgency = "LOW"
            strategy = "You can provide detailed analysis within your allocated tokens."
        
        # Role-specific guidance
        role_specific_guidance = {
            AgentRole.HYPOTHESIS: "Focus on top 2-3 diagnoses with probabilities. Prioritize summary over detailed pathophysiology.",
            AgentRole.TEST_CHOOSER: "Recommend 1-2 highest-yield tests. Focus on which hypotheses they'll help differentiate.",
            AgentRole.CHALLENGER: "Identify 1-2 most critical biases or alternative diagnoses. Be direct and specific.",
            AgentRole.STEWARDSHIP: "Focus on cost-effectiveness assessment. Recommend cheaper alternatives where applicable.",
            AgentRole.CHECKLIST: "Provide concise quality check. Flag critical issues only.",
            AgentRole.CONSENSUS: "Function calling enforces structure. Focus on clear reasoning.",
            AgentRole.GATEKEEPER: "Provide specific clinical findings. Be factual and complete but not verbose.",
            AgentRole.JUDGE: "Provide score and focused justification. Be systematic but concise."
        }.get(agent_role, "Be concise and focused.")
        
        guidance = f"""
[TOKEN MANAGEMENT - {urgency} PRIORITY]
Input: {input_tokens} tokens | Your Output Limit: {max_output_tokens} tokens | Total: {total_tokens} tokens
Strategy: {strategy}
Role Focus: {role_specific_guidance}

IMPORTANT: Adjust your response length and detail level based on this guidance. Prioritize the most critical information for your role.
"""
        
        return guidance

    def _init_agents(self) -> None:
        """Initializes all required agents with their specific roles and prompts."""
        
        # Define the structured output tool for consensus decisions
        consensus_tool = {
            "type": "function",
            "function": {
                "name": "make_consensus_decision",
                "description": "Make a structured consensus decision for the next diagnostic action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["ask", "test", "diagnose"],
                            "description": "The type of action to perform"
                        },
                        "content": {
                            "type": "string",
                            "description": "The specific content of the action (question, test name, or diagnosis)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "The detailed reasoning behind this decision, synthesizing panel input"
                        }
                    },
                    "required": ["action_type", "content", "reasoning"]
                }
            }
        }
        
        # Define structured output tool for differential diagnosis
        hypothesis_tool = {
            "type": "function",
            "function": {
                "name": "update_differential_diagnosis",
                "description": "Update the differential diagnosis with structured probabilities and reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "One-sentence summary of primary diagnostic conclusion and confidence"
                        },
                        "differential_diagnoses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "diagnosis": {"type": "string", "description": "The diagnosis name"},
                                    "probability": {"type": "number", "minimum": 0, "maximum": 1, "description": "Probability as decimal (0.0-1.0)"},
                                    "rationale": {"type": "string", "description": "Brief rationale for this diagnosis"}
                                },
                                "required": ["diagnosis", "probability", "rationale"]
                            },
                            "minItems": 2,
                            "maxItems": 5,
                            "description": "Top 2-5 differential diagnoses with probabilities"
                        },
                        "key_evidence": {
                            "type": "string",
                            "description": "Key supporting evidence for leading hypotheses"
                        },
                        "contradictory_evidence": {
                            "type": "string", 
                            "description": "Critical contradictory evidence that must be addressed"
                        }
                    },
                    "required": ["summary", "differential_diagnoses", "key_evidence"]
                }
            }
        }
        
        self.agents = {}
        for role in AgentRole:
            if role == AgentRole.CONSENSUS:
                # Use function calling for consensus agent to ensure structured output
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    tools_list_dictionary=[consensus_tool],  # swarms expects tools_list_dictionary
                    tool_choice="auto",  # Let the model choose to use the tool
                    print_on=True,
                    max_tokens=self._get_agent_max_tokens(role),
                )
            elif role == AgentRole.HYPOTHESIS:
                # Use function calling for hypothesis agent to ensure structured differential
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    tools_list_dictionary=[hypothesis_tool],
                    tool_choice="auto",
                    print_on=True,
                    max_tokens=self._get_agent_max_tokens(role),
                )
            else:
                # Regular agents without function calling
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    output_type="str",
                    print_on=True,
                    max_tokens=self._get_agent_max_tokens(role),
                )
        
        logger.info(
            f"👥 {len(self.agents)} virtual physician agents initialized and ready for consultation"
        )

    def _get_dynamic_context(self, role: AgentRole, case_state: CaseState) -> str:
        """Generate dynamic context for agents based on current situation - addresses Category 4.2"""
        remaining_budget = self.initial_budget - case_state.cumulative_cost
        
        # Calculate confidence from differential diagnosis
        max_confidence = max(case_state.differential_diagnosis.values()) if case_state.differential_diagnosis else 0
        
        context = ""
        
        if role == AgentRole.STEWARDSHIP and remaining_budget < 1000:
            context = f"""
**SITUATIONAL CONTEXT: URGENT**
The remaining budget is critically low (${remaining_budget}). All recommendations must be focused on maximum cost-effectiveness. Veto any non-essential or high-cost tests.
"""
        
        elif role == AgentRole.HYPOTHESIS and max_confidence > 0.75:
            context = f"""
**SITUATIONAL CONTEXT: FINAL STAGES**
The panel is converging on a diagnosis (current max confidence: {max_confidence:.0%}). Your primary role now is to confirm the leading hypothesis or state what single piece of evidence is needed to reach >85% confidence.
"""
        
        elif role == AgentRole.CONSENSUS and case_state.iteration > 5:
            context = f"""
**SITUATIONAL CONTEXT: EXTENDED CASE**
This case has gone through {case_state.iteration} iterations. Focus on decisive actions that will lead to a definitive diagnosis rather than additional exploratory steps.
"""
        
        return context

    def _get_prompt_for_role(self, role: AgentRole, case_state: CaseState = None) -> str:
        """Returns the system prompt for a given agent role with dynamic context."""
        
        # Add dynamic context if case_state is provided
        dynamic_context = ""
        if case_state:
            dynamic_context = self._get_dynamic_context(role, case_state)
        
        base_prompts = {
            AgentRole.HYPOTHESIS: f"""
            {dynamic_context}
            
            You are Dr. Hypothesis, a specialist in maintaining differential diagnoses. Your role is critical to the diagnostic process.

            CORE RESPONSIBILITIES:
            - Maintain a probability-ranked differential diagnosis with the top 3-5 most likely conditions
            - Update probabilities using Bayesian reasoning after each new finding
            - Consider both common and rare diseases appropriate to the clinical context
            - Explicitly track how new evidence changes your diagnostic thinking
            - Provide comprehensive analysis with detailed clinical reasoning

            APPROACH:
            1. Start with the most likely diagnoses based on presenting symptoms
            2. For each new piece of evidence, consider:
               - How it supports or refutes each hypothesis
               - Whether it suggests new diagnoses to consider
               - How it changes the relative probabilities
            3. Always explain your Bayesian reasoning clearly
            4. Consider epidemiology, pathophysiology, and clinical patterns

            **IMPORTANT: You MUST use the update_differential_diagnosis function to provide your structured analysis.**
            
            Use the function to provide:
            - A one-sentence summary of your primary diagnostic conclusion and confidence level
            - Your top 2-5 differential diagnoses with probability estimates (as decimals: 0.0-1.0)
            - Brief rationale for each diagnosis
            - Key supporting evidence for leading hypotheses  
            - Critical contradictory evidence that must be addressed

            Remember: Your differential drives the entire diagnostic process. Provide clear probabilities and reasoning.
            """,
            
            AgentRole.TEST_CHOOSER: f"""
            {dynamic_context}
            
            You are Dr. Test-Chooser, a specialist in diagnostic test selection and information theory.

            CORE RESPONSIBILITIES:
            - Select up to 3 diagnostic tests per round that maximally discriminate between leading hypotheses
            - Optimize for information value, not just clinical reasonableness
            - Consider test characteristics: sensitivity, specificity, positive/negative predictive values
            - Balance diagnostic yield with patient burden and resource utilization
            - Provide comprehensive test selection rationale

            SELECTION CRITERIA:
            1. Information Value: How much will this test change diagnostic probabilities?
            2. Discriminatory Power: How well does it distinguish between competing hypotheses?
            3. Clinical Impact: Will the result meaningfully alter management?
            4. Sequential Logic: What should we establish first before ordering more complex tests?
            5. Cost-effectiveness and patient safety considerations

            APPROACH:
            - For each proposed test, explicitly state which hypotheses it will help confirm or exclude
            - Consider both positive and negative results and their implications
            - Think about test sequences (e.g., basic labs before advanced imaging)
            - Avoid redundant tests that won't add new information
            - Consider pre-test probability and post-test probability calculations

            OUTPUT FORMAT (You have a response limit of {self._get_agent_max_tokens(AgentRole.TEST_CHOOSER)} tokens - prioritize actionable recommendations):
            
            **SUMMARY FIRST:** Lead with your single most recommended test and why it's the highest priority.
            
            **DETAILED RECOMMENDATIONS (up to 3 tests):**
            For each test:
            - Test name (be specific and accurate)
            - Primary hypotheses it will help evaluate
            - Expected information gain
            - How results will change management
            - Cost-effectiveness assessment
            - Timing rationale

            Focus on tests that will most efficiently narrow the differential diagnosis.
            """,
            
            AgentRole.CHALLENGER: f"""
            {dynamic_context}
            
            You are Dr. Challenger, the critical thinking specialist and devil's advocate.

            CORE RESPONSIBILITIES:
            - Identify and challenge cognitive biases in the diagnostic process
            - Highlight contradictory evidence that might be overlooked
            - Propose alternative hypotheses and falsifying tests
            - Guard against premature diagnostic closure
            - Provide comprehensive critical analysis

            COGNITIVE BIASES TO WATCH FOR:
            1. Anchoring: Over-reliance on initial impressions
            2. Confirmation bias: Seeking only supporting evidence
            3. Availability bias: Overestimating probability of recently seen conditions
            4. Representativeness: Ignoring base rates and prevalence
            5. Search satisficing: Stopping at "good enough" explanations
            6. Attribution errors and hindsight bias

            YOUR APPROACH:
            - Ask "What else could this be?" and "What doesn't fit?"
            - Challenge assumptions and look for alternative explanations
            - Propose tests that could disprove the leading hypothesis
            - Consider rare diseases when common ones don't fully explain the picture
            - Advocate for considering multiple conditions simultaneously
            - Look for inconsistencies in the clinical presentation

            OUTPUT FORMAT (You have a response limit of {self._get_agent_max_tokens(AgentRole.CHALLENGER)} tokens - focus on the most critical challenges):
            
            **SUMMARY FIRST:** State your primary concern with the current diagnostic approach in one sentence.
            
            **CRITICAL CHALLENGES:**
            - Most significant bias identified in current reasoning
            - Key evidence that contradicts leading hypotheses
            - Most important alternative diagnosis to consider
            - Essential test to falsify current assumptions
            - Highest priority red flag or safety concern
            - Most critical gap in current approach

            Be constructively critical - focus on the challenges that most impact diagnostic accuracy.
            """,
            
            AgentRole.STEWARDSHIP: f"""
            {dynamic_context}
            
            You are Dr. Stewardship, the resource optimization and cost-effectiveness specialist.

            CORE RESPONSIBILITIES:
            - Enforce cost-conscious, high-value care
            - Advocate for cheaper alternatives when diagnostically equivalent
            - Challenge low-yield, expensive tests
            - Balance diagnostic thoroughness with resource stewardship
            - Provide comprehensive cost-benefit analysis

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
            - Are there outpatient vs. inpatient cost considerations?

            YOUR APPROACH:
            - Review all proposed tests for necessity and value
            - Suggest cost-effective alternatives with rationale
            - Question tests that don't clearly advance diagnosis
            - Advocate for asking questions before ordering expensive tests
            - Consider the cumulative cost burden and budget constraints
            - Analyze cost per unit of diagnostic information gained

            OUTPUT FORMAT (Use full token allocation for detailed analysis):
            - Assessment of proposed tests (high/moderate/low/no value) with detailed reasoning
            - Specific cost-effective alternatives with cost comparisons
            - Questions that might obviate need for testing
            - Recommended modifications to testing strategy
            - Cumulative cost considerations and budget impact
            - Value-based care recommendations
            - Analysis of diagnostic yield vs. cost for each proposed intervention

            Your goal: Maximum diagnostic accuracy at minimum necessary cost while maintaining high-quality care.
            """,
            
            AgentRole.CHECKLIST: f"""
            {dynamic_context}
            
            You are Dr. Checklist, the quality assurance and consistency specialist.

            CORE RESPONSIBILITIES:
            - Perform comprehensive quality control on all panel deliberations
            - Ensure test names are valid and properly specified
            - Check internal consistency of reasoning across panel members
            - Flag logical errors or contradictions in the diagnostic approach
            - Provide systematic quality assessment

            QUALITY CHECKS:
            1. Test Validity: Are proposed tests real and properly named?
            2. Logical Consistency: Do the recommendations align with the differential?
            3. Evidence Integration: Are all findings being considered appropriately?
            4. Process Adherence: Is the panel following proper diagnostic methodology?
            5. Safety Checks: Are any critical possibilities being overlooked?
            6. Completeness: Is the diagnostic workup comprehensive?

            SPECIFIC VALIDATIONS:
            - Test names match standard medical terminology
            - Proposed tests are appropriate for the clinical scenario
            - No contradictions between different panel members' reasoning
            - All significant findings are being addressed
            - No gaps in the diagnostic logic
            - Proper consideration of differential diagnosis breadth

            OUTPUT FORMAT (Use full token allocation for comprehensive analysis):
            - Detailed validation summary (✓ Clear / ⚠ Issues noted)
            - Any test name corrections needed with proper terminology
            - Logical inconsistencies identified with specific examples
            - Missing considerations or gaps in reasoning
            - Process improvement suggestions with rationale
            - Safety concerns or red flags that need immediate attention
            - Systematic review of diagnostic approach quality

            Keep your feedback comprehensive and detailed. Flag any issues that could compromise diagnostic quality or patient safety.
            """,
            
            AgentRole.CONSENSUS: f"""
            {dynamic_context}
            
            You are the Consensus Coordinator, responsible for synthesizing the virtual panel's expertise into a single, optimal decision.

            CORE RESPONSIBILITIES:
            - Integrate input from Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, and Dr. Checklist
            - Decide on the single best next action: 'ask', 'test', or 'diagnose'
            - Balance competing priorities: accuracy, cost, efficiency, and thoroughness
            - Ensure the chosen action advances the diagnostic process optimally

            **PRIORITIZED DECISION FRAMEWORK:**
            Use the following prioritized framework to make your decision:

            1. **Certainty Threshold:** If Dr. Hypothesis's leading diagnosis has confidence >85% AND Dr. Challenger raises no major objections, your action MUST be `diagnose`.
            2. **Address Red Flags:** If Dr. Challenger identifies a critical bias or contradictory evidence, your next action MUST be a test or question that directly addresses that challenge.
            3. **High-Value Information:** Otherwise, select the test from Dr. Test-Chooser that offers the highest information gain.
            4. **Cost Optimization:** Before finalizing a test, check Dr. Stewardship's input. If a diagnostically equivalent but cheaper alternative is available, select it.
            5. **Default to Questions:** If no test meets the criteria or the budget is a major concern, select the most pertinent question to ask.

            **IMPORTANT: You MUST use the make_consensus_decision function to provide your structured response. Call this function with the appropriate action_type, content, and reasoning parameters.**
            
            For action_type "ask": content should be specific patient history or physical exam questions
            For action_type "test": content should be properly named diagnostic tests (up to 3)
            For action_type "diagnose": content should be the complete, specific final diagnosis

            Make the decision that best advances accurate, cost-effective diagnosis. Use comprehensive reasoning that synthesizes all panel input and cites the specific decision framework step you're following.
            """,
            
            AgentRole.GATEKEEPER: f"""
            {dynamic_context}
            
            You are the Gatekeeper, the clinical information oracle with complete access to the patient case file.

            CORE RESPONSIBILITIES:
            - Provide objective, specific clinical findings when explicitly requested
            - Serve as the authoritative source for all patient information
            - Generate realistic synthetic findings for tests not in the original case
            - Maintain clinical realism while preventing information leakage
            - Provide comprehensive, detailed responses

            RESPONSE PRINCIPLES:
            1. OBJECTIVITY: Provide only factual findings, never interpretations or impressions
            2. SPECIFICITY: Give precise, detailed results when tests are properly ordered
            3. REALISM: Ensure all responses reflect realistic clinical scenarios
            4. NO HINTS: Never provide diagnostic clues or suggestions
            5. CONSISTENCY: Maintain coherence across all provided information
            6. COMPLETENESS: Provide thorough, detailed responses

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
            - Consider normal variations and expected findings

            RESPONSE FORMAT (Use full token allocation for detailed responses):
            - Direct, clinical language with comprehensive detail
            - Specific measurements with reference ranges when applicable
            - Clear organization of findings with systematic presentation
            - Professional medical terminology with full descriptions
            - Complete documentation as would appear in medical records

            Your role is crucial: provide complete, accurate clinical information while maintaining the challenge of the diagnostic process. Use your full token allocation to provide comprehensive, detailed clinical information.
            """,
            
            AgentRole.JUDGE: f"""
            {dynamic_context}
            
            You are the Judge, the diagnostic accuracy evaluation specialist.

            CORE RESPONSIBILITIES:
            - Evaluate candidate diagnoses against ground truth using a rigorous clinical rubric
            - Provide fair, consistent scoring based on clinical management implications
            - Consider diagnostic substance over terminology differences
            - Account for acceptable medical synonyms and equivalent formulations
            - Provide comprehensive evaluation reasoning

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

            OUTPUT FORMAT (Use full token allocation for comprehensive evaluation):
            - Score (1-5) with clear label and detailed justification
            - Comprehensive reasoning referencing specific rubric criteria
            - Detailed explanation of how diagnosis would affect clinical management
            - Note any acceptable medical synonyms or equivalent terminology
            - Analysis of diagnostic accuracy and clinical implications
            - Systematic comparison with ground truth diagnosis

            Maintain high standards while recognizing legitimate diagnostic variability in medical practice. Provide comprehensive, detailed evaluation.
            """,
        }
        
        # Use existing prompts for other roles, just add dynamic context
        if role not in base_prompts:
            return dynamic_context + self._get_original_prompt_for_role(role)
        
        return base_prompts[role]

    def _get_original_prompt_for_role(self, role: AgentRole) -> str:
        """Returns original system prompts for roles not yet updated"""
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

    def _parse_json_response(self, response: str, retry_count: int = 0) -> Dict[str, Any]:
        """Safely parses a JSON string with retry logic - addresses Category 3.2"""
        try:
            # Handle agent response wrapper - extract actual content
            if isinstance(response, dict):
                # Handle swarms Agent response format
                if 'role' in response and 'content' in response:
                    response = response['content']
                elif 'content' in response:
                    response = response['content']
                else:
                    # Try to extract any string value from dict
                    response = str(response)
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                # Convert to string if it's some other type
                response = str(response)

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

                # Look for JSON pattern in the text - more comprehensive regex
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)

                for match in matches:
                    try:
                        parsed = json.loads(match)
                        # Validate that it has the expected action structure
                        if isinstance(parsed, dict) and 'action_type' in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue

                # Direct parsing attempt as fallback
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # --- Fallback Sanitization ---
                    # Attempt to strip any leading table/frame characters (e.g., │, ╭, ╰) that may wrap each line
                    try:
                        # Extract everything between the first '{' and last '}'
                        start_curly = response.index('{')
                        end_curly = response.rindex('}')
                        candidate = response[start_curly:end_curly + 1]
                        sanitized_lines = []
                        for line in candidate.splitlines():
                            # Remove common frame characters and leading whitespace
                            line = line.lstrip('│|╭╰╯├─┤ ').rstrip('│|╭╰╯├─┤ ')
                            sanitized_lines.append(line)
                        candidate_clean = '\n'.join(sanitized_lines)
                        return json.loads(candidate_clean)
                    except Exception as inner_e:
                        # Still failing, raise original error to trigger retry logic
                        try:
                            # --- Ultimate Fallback: Regex extraction ---
                            import re
                            atype = re.search(r'"action_type"\s*:\s*"(ask|test|diagnose)"', response, re.IGNORECASE)
                            content_match = re.search(r'"content"\s*:\s*"([^"]+?)"', response, re.IGNORECASE | re.DOTALL)
                            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+?)"', response, re.IGNORECASE | re.DOTALL)
                            if atype and content_match and reasoning_match:
                                return {
                                    "action_type": atype.group(1).lower(),
                                    "content": content_match.group(1).strip(),
                                    "reasoning": reasoning_match.group(1).strip(),
                                }
                        except Exception:
                            pass
                        raise e

        except (
            json.JSONDecodeError,
            IndexError,
            AttributeError,
        ) as e:
            logger.error(f"Failed to parse JSON response. Error: {e}")
            logger.debug(
                f"Response content: {response[:500]}..."
            )  # Log first 500 chars
            
            # Return the error for potential retry instead of immediately falling back
            raise e
    
    def _parse_json_with_retry(self, consensus_agent: Agent, consensus_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Parse JSON with retry logic for robustness - addresses Category 3.2"""
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    response = consensus_agent.run(consensus_prompt)
                else:
                    # Retry with error feedback
                    retry_prompt = f"""
{consensus_prompt}

**CRITICAL: RETRY REQUIRED - ATTEMPT {attempt + 1}**
Your previous response could not be parsed as JSON. You MUST respond with ONLY a valid JSON object in exactly this format:

{{
    "action_type": "ask" | "test" | "diagnose",
    "content": "your content here",
    "reasoning": "your reasoning here"
}}

Do NOT include any other text, markdown formatting, or explanations. Only the raw JSON object.
NO SYSTEM MESSAGES, NO WRAPPER FORMAT. JUST THE JSON.
"""
                    response = consensus_agent.run(retry_prompt)
                
                # Handle different response types from swarms Agent
                response_text = ""
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict):
                    # Handle swarms Agent response wrapper
                    if 'role' in response and 'content' in response:
                        response_text = response['content']
                    elif 'content' in response:
                        response_text = response['content']
                    else:
                        response_text = str(response)
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)
                
                # Log the response for debugging
                logger.debug(f"Parsing attempt {attempt + 1}, response type: {type(response)}")
                logger.debug(f"Response content preview: {str(response_text)[:200]}...")
                
                return self._parse_json_response(response_text, attempt)
                
            except Exception as e:
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    # Final fallback after all retries
                    logger.error("All JSON parsing attempts failed, using fallback")
                    return {
                        "action_type": "ask",
                        "content": "Could you please clarify the next best step? The previous analysis was inconclusive.",
                        "reasoning": f"Fallback due to JSON parsing error after {max_retries + 1} attempts.",
                    }
        
        # Should never reach here, but just in case
        return {
            "action_type": "ask",
            "content": "Please provide more information about the patient's condition.",
            "reasoning": "Unexpected fallback in JSON parsing.",
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

    def _run_panel_deliberation(self, case_state: CaseState) -> Action:
        """Orchestrates one round of structured debate among the virtual panel - addresses Category 1.1"""
        logger.info(
            "🩺 Virtual medical panel deliberation commenced - analyzing patient case"
        )
        logger.debug(
            "Panel members: Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, Dr. Checklist"
        )

        # Initialize structured deliberation state instead of conversational chaining
        deliberation_state = DeliberationState()
        
        # Prepare concise case context for each agent (token-optimized)
        remaining_budget = self.initial_budget - case_state.cumulative_cost
        budget_status = (
            "EXCEEDED"
            if remaining_budget < 0
            else f"${remaining_budget:,}"
        )

        # Full context - let agents self-regulate based on token guidance
        base_context = f"""
=== DIAGNOSTIC CASE STATUS - ROUND {case_state.iteration} ===

INITIAL PRESENTATION:
{case_state.initial_vignette}

EVIDENCE GATHERED:
{case_state.summarize_evidence()}

CURRENT STATE:
- Tests Performed: {', '.join(case_state.tests_performed) if case_state.tests_performed else 'None'}
- Questions Asked: {len(case_state.questions_asked)}
- Cumulative Cost: ${case_state.cumulative_cost:,}
- Remaining Budget: {budget_status}
- Mode: {self.mode}
        """

        # Check mode-specific constraints
        if self.mode == "instant":
            # For instant mode, skip deliberation and go straight to diagnosis
            action_dict = {
                "action_type": "diagnose",
                "content": case_state.get_leading_diagnosis(),
                "reasoning": (
                    "Instant diagnosis mode - providing immediate assessment based on initial presentation"
                ),
            }
            return Action(**action_dict)

        # Check for stagnation before running deliberation
        stagnation_detected = False
        if len(case_state.last_actions) >= 2:
            last_action = case_state.last_actions[-1]
            stagnation_detected = case_state.is_stagnating(last_action)
            deliberation_state.stagnation_detected = stagnation_detected
            if stagnation_detected:
                logger.warning("🔄 Stagnation detected - will force different action")

        # Generate dynamic situational context for all agents
        deliberation_state.situational_context = self._generate_situational_context(case_state, remaining_budget)

        # Run each specialist agent in parallel-like fashion with structured output
        # Each agent gets the same base context plus their role-specific dynamic prompt
        try:
            # Dr. Hypothesis - Differential diagnosis and probability assessment
            logger.info("🧠 Dr. Hypothesis analyzing differential diagnosis...")
            hypothesis_prompt = self._get_prompt_for_role(AgentRole.HYPOTHESIS, case_state) + "\n\n" + base_context
            hypothesis_response = self._safe_agent_run(
                self.agents[AgentRole.HYPOTHESIS], hypothesis_prompt, agent_role=AgentRole.HYPOTHESIS
            )
            
            # Update case state with new differential (supports both function calls and text)
            self._update_differential_from_hypothesis(case_state, hypothesis_response)
            
            # Store the analysis for deliberation state (convert to text format for other agents)
            if hasattr(hypothesis_response, 'content'):
                deliberation_state.hypothesis_analysis = hypothesis_response.content
            else:
                deliberation_state.hypothesis_analysis = str(hypothesis_response)

            # Dr. Test-Chooser - Information value optimization
            logger.info("🔬 Dr. Test-Chooser selecting optimal tests...")
            test_chooser_prompt = self._get_prompt_for_role(AgentRole.TEST_CHOOSER, case_state) + "\n\n" + base_context
            if self.mode == "question_only":
                test_chooser_prompt += "\n\nIMPORTANT: This is QUESTION-ONLY mode. You may ONLY recommend patient questions, not diagnostic tests."
            deliberation_state.test_chooser_analysis = self._safe_agent_run(
                self.agents[AgentRole.TEST_CHOOSER], test_chooser_prompt, agent_role=AgentRole.TEST_CHOOSER
            )

            # Dr. Challenger - Bias identification and alternative hypotheses
            logger.info("🤔 Dr. Challenger challenging assumptions...")
            challenger_prompt = self._get_prompt_for_role(AgentRole.CHALLENGER, case_state) + "\n\n" + base_context
            deliberation_state.challenger_analysis = self._safe_agent_run(
                self.agents[AgentRole.CHALLENGER], challenger_prompt, agent_role=AgentRole.CHALLENGER
            )

            # Dr. Stewardship - Cost-effectiveness analysis
            logger.info("💰 Dr. Stewardship evaluating cost-effectiveness...")
            stewardship_prompt = self._get_prompt_for_role(AgentRole.STEWARDSHIP, case_state) + "\n\n" + base_context
            if self.enable_budget_tracking:
                stewardship_prompt += f"\n\nBUDGET TRACKING ENABLED - Current cost: ${case_state.cumulative_cost}, Remaining: ${remaining_budget}"
            deliberation_state.stewardship_analysis = self._safe_agent_run(
                self.agents[AgentRole.STEWARDSHIP], stewardship_prompt, agent_role=AgentRole.STEWARDSHIP
            )

            # Dr. Checklist - Quality assurance
            logger.info("✅ Dr. Checklist performing quality control...")
            checklist_prompt = self._get_prompt_for_role(AgentRole.CHECKLIST, case_state) + "\n\n" + base_context
            deliberation_state.checklist_analysis = self._safe_agent_run(
                self.agents[AgentRole.CHECKLIST], checklist_prompt, agent_role=AgentRole.CHECKLIST
            )

            # Consensus Coordinator - Final decision synthesis using structured state
            logger.info("🤝 Consensus Coordinator synthesizing panel decision...")
            
            # Generate the structured consensus prompt
            consensus_prompt = deliberation_state.to_consensus_prompt()
            
            # Add mode-specific constraints to consensus
            if self.mode == "budgeted" and remaining_budget <= 0:
                consensus_prompt += "\n\nBUDGET CONSTRAINT: Budget exceeded - must either ask questions or provide final diagnosis."

            # Use function calling with retry logic for robust structured output
            action_dict = self._get_consensus_with_retry(consensus_prompt)

            # Validate action based on mode constraints
            action = Action(**action_dict)
            
            # Apply mode-specific validation and corrections
            action = self._validate_and_correct_action(action, case_state, remaining_budget)

            return action

        except Exception as e:
            logger.error(f"Error during panel deliberation: {e}")
            # Fallback action
            return Action(
                action_type="ask",
                content="Could you please provide more information about the patient's current condition?",
                reasoning=f"Fallback due to panel deliberation error: {str(e)}",
            )
    
    def _generate_situational_context(self, case_state: CaseState, remaining_budget: int) -> str:
        """Generate dynamic situational context based on current case state - addresses Category 4.2"""
        context_parts = []
        
        # Budget-related context
        if remaining_budget < 1000:
            context_parts.append(f"URGENT: Remaining budget critically low (${remaining_budget}). Focus on cost-effective actions.")
        elif remaining_budget < 2000:
            context_parts.append(f"WARNING: Budget running low (${remaining_budget}). Prioritize high-value tests.")
        
        # Diagnostic confidence context
        max_confidence = case_state.get_max_confidence()
        if max_confidence > 0.85:
            context_parts.append(f"FINAL STAGES: High confidence diagnosis available ({max_confidence:.0%}). Consider definitive action.")
        elif max_confidence > 0.70:
            context_parts.append(f"CONVERGING: Moderate confidence in leading diagnosis ({max_confidence:.0%}). Focus on confirmation.")
        
        # Iteration context
        if case_state.iteration > 7:
            context_parts.append(f"EXTENDED CASE: {case_state.iteration} rounds completed. Move toward decisive action.")
        elif case_state.iteration > 5:
            context_parts.append(f"PROLONGED: {case_state.iteration} rounds. Avoid further exploratory steps unless critical.")
        
        # Test/cost context
        if len(case_state.tests_performed) > 5:
            context_parts.append("EXTENSIVE TESTING: Many tests completed. Focus on synthesis rather than additional testing.")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _update_differential_from_hypothesis(self, case_state: CaseState, hypothesis_response):
        """Extract and update differential diagnosis from Dr. Hypothesis analysis - now supports both function calls and text"""
        try:
            # Try to extract structured data from function call first
            if hasattr(hypothesis_response, '__dict__') or isinstance(hypothesis_response, dict):
                structured_data = self._extract_function_call_output(hypothesis_response)
                
                # Check if we got structured differential data
                if "differential_diagnoses" in structured_data:
                    # Update case state with structured data
                    new_differential = {}
                    for dx in structured_data["differential_diagnoses"]:
                        new_differential[dx["diagnosis"]] = dx["probability"]
                    
                    case_state.update_differential(new_differential)
                    
                    # Update the main differential for backward compatibility
                    summary = structured_data.get("summary", "Differential diagnosis updated")
                    dx_text = f"{summary}\n\nTop Diagnoses:\n"
                    for dx in structured_data["differential_diagnoses"]:
                        dx_text += f"- {dx['diagnosis']}: {dx['probability']:.0%} - {dx['rationale']}\n"
                    
                    if "key_evidence" in structured_data:
                        dx_text += f"\nKey Evidence: {structured_data['key_evidence']}"
                    if "contradictory_evidence" in structured_data:
                        dx_text += f"\nContradictory Evidence: {structured_data['contradictory_evidence']}"
                        
                    self.differential_diagnosis = dx_text
                    logger.debug(f"Updated differential from function call: {new_differential}")
                    return
            
            # Fallback to text-based extraction
            hypothesis_text = str(hypothesis_response)
            if hasattr(hypothesis_response, 'content'):
                hypothesis_text = hypothesis_response.content
                
            # Simple extraction - look for percentage patterns in the text
            import re
            
            # Update the main differential diagnosis for backward compatibility
            self.differential_diagnosis = hypothesis_text
            
            # Try to extract structured probabilities
            # Look for patterns like "Diagnosis: 85%" or "Disease (70%)"
            percentage_pattern = r'([A-Za-z][^:(\n]*?)[\s:]*[\(]?(\d{1,3})%[\)]?'
            matches = re.findall(percentage_pattern, hypothesis_text)
            
            new_differential = {}
            for match in matches:
                diagnosis = match[0].strip().rstrip(':-()').strip()
                probability = float(match[1]) / 100.0
                if 0 <= probability <= 1.0 and len(diagnosis) > 3:  # Basic validation
                    new_differential[diagnosis] = probability
            
            if new_differential:
                case_state.update_differential(new_differential)
                logger.debug(f"Updated differential from text parsing: {new_differential}")
                
        except Exception as e:
            logger.debug(f"Could not extract structured differential: {e}")
            # Still update the text version for display
            hypothesis_text = str(hypothesis_response)
            if hasattr(hypothesis_response, 'content'):
                hypothesis_text = hypothesis_response.content
            self.differential_diagnosis = hypothesis_text
    
    def _validate_and_correct_action(self, action: Action, case_state: CaseState, remaining_budget: int) -> Action:
        """Validate and correct actions based on mode constraints and context"""
        
        # Mode-specific validations
        if self.mode == "question_only" and action.action_type == "test":
            logger.warning("Test ordering attempted in question-only mode, converting to ask action")
            action.action_type = "ask"
            action.content = "Can you provide more details about the patient's symptoms and history?"
            action.reasoning = "Mode constraint: question-only mode active"
        
        if self.mode == "budgeted" and action.action_type == "test" and remaining_budget <= 0:
            logger.warning("Test ordering attempted with insufficient budget, converting to diagnose action")
            action.action_type = "diagnose"
            action.content = case_state.get_leading_diagnosis()
            action.reasoning = "Budget constraint: insufficient funds for additional testing"
        
        # Stagnation handling - ensure we have a valid diagnosis
        if case_state.is_stagnating(action):
            logger.warning("Stagnation detected, forcing diagnostic decision")
            action.action_type = "diagnose"
            leading_diagnosis = case_state.get_leading_diagnosis()
            # Ensure the diagnosis is meaningful, not corrupted
            if leading_diagnosis == "No diagnosis formulated" or len(leading_diagnosis) < 10 or any(char in leading_diagnosis for char in ['x10^9', '–', '40–']):
                # Use a fallback diagnosis based on the case context
                action.content = "Unable to establish definitive diagnosis - further evaluation needed"
            else:
                action.content = leading_diagnosis
            action.reasoning = "Forced diagnosis due to detected stagnation in diagnostic process"
        
        # High confidence threshold
        if action.action_type != "diagnose" and case_state.get_max_confidence() > 0.90:
            logger.info("Very high confidence reached, recommending diagnosis")
            action.action_type = "diagnose"
            action.content = case_state.get_leading_diagnosis()
            action.reasoning = "High confidence threshold reached - proceeding to final diagnosis"
        
        return action

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

        response = self._safe_agent_run(gatekeeper, prompt, agent_role=AgentRole.GATEKEEPER)
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
        
        You must provide your evaluation in exactly this format:
        Score: [number from 1-5]
        Justification: [detailed reasoning for the score]
        """
        response = self._safe_agent_run(judge, prompt, agent_role=AgentRole.JUDGE)
        
        # Handle different response types from swarms Agent
        response_text = ""
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, dict):
            if 'role' in response and 'content' in response:
                response_text = response['content']
            elif 'content' in response:
                response_text = response['content']
            else:
                response_text = str(response)
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)

        # Enhanced parsing for demonstration; a more robust solution would use structured output.
        try:
            # Look for score patterns
            import re
            
            # Try multiple score patterns
            score_patterns = [
                r"Score:\s*(\d+(?:\.\d+)?)",
                r"Score\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)/5",
                r"Score.*?(\d+(?:\.\d+)?)",
            ]
            
            score = 0.0
            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    break
            
            # Extract reasoning
            reasoning_patterns = [
                r"Justification:\s*(.+?)(?:\n\n|\Z)",
                r"Reasoning:\s*(.+?)(?:\n\n|\Z)",
                r"Explanation:\s*(.+?)(?:\n\n|\Z)",
            ]
            
            reasoning = "Could not parse judge's reasoning."
            for pattern in reasoning_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    break
            
            # If no specific reasoning found, use the whole response after score
            if reasoning == "Could not parse judge's reasoning." and score > 0:
                # Try to extract everything after the score
                score_match = re.search(r"Score:?\s*\d+(?:\.\d+)?", response_text, re.IGNORECASE)
                if score_match:
                    reasoning = response_text[score_match.end():].strip()
                    # Clean up common prefixes
                    reasoning = re.sub(r"^(Justification|Reasoning|Explanation):\s*", "", reasoning, flags=re.IGNORECASE)
                
            # Final fallback - use the whole response if we have a score
            if reasoning == "Could not parse judge's reasoning." and score > 0:
                reasoning = response_text
                
        except (IndexError, ValueError, AttributeError) as e:
            logger.error(f"Error parsing judge response: {e}")
            score = 0.0
            reasoning = f"Could not parse judge's response: {str(e)}"

        logger.info(f"Judge evaluation: Score={score}, Reasoning preview: {reasoning[:100]}...")
        return {"score": score, "reasoning": reasoning}

    def run(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
    ) -> DiagnosisResult:
        """
        Executes the full sequential diagnostic process with structured state management.

        Args:
            initial_case_info (str): The initial abstract of the case.
            full_case_details (str): The complete case file for the Gatekeeper.
            ground_truth_diagnosis (str): The correct final diagnosis for evaluation.

        Returns:
            DiagnosisResult: An object containing the final diagnosis, evaluation, cost, and history.
        """
        start_time = time.time()
        
        # Initialize structured case state
        case_state = CaseState(initial_vignette=initial_case_info)
        case_state.cumulative_cost = self.physician_visit_cost  # Add initial visit cost
        self.cumulative_cost = case_state.cumulative_cost
        
        # Store for potential use by other methods
        self.case_state = case_state
        
        # Add to conversation for history tracking
        self.conversation.add(
            "Gatekeeper",
            f"Initial Case Information: {initial_case_info}",
        )
        case_state.add_evidence(f"Initial presentation: {initial_case_info}")

        logger.info(
            f"Initial physician visit cost: ${self.physician_visit_cost}"
        )

        final_diagnosis = None
        iteration_count = 0

        for i in range(self.max_iterations):
            iteration_count = i + 1
            case_state.iteration = iteration_count
            
            logger.info(
                f"--- Starting Diagnostic Loop {iteration_count}/{self.max_iterations} ---"
            )
            logger.info(
                f"Current cost: ${case_state.cumulative_cost:,} | Remaining budget: ${self.initial_budget - case_state.cumulative_cost:,}"
            )

            try:
                # Panel deliberates to decide on the next action using structured state
                action = self._run_panel_deliberation(case_state)
                logger.info(
                    f"⚕️ Panel decision: {action.action_type.upper()} -> {action.content}"
                )
                logger.info(
                    f"💭 Medical reasoning: {action.reasoning}"
                )

                # Add action to case state for stagnation detection
                case_state.add_action(action)

                if action.action_type == "diagnose":
                    final_diagnosis = action.content
                    logger.info(
                        f"Final diagnosis proposed: {final_diagnosis}"
                    )
                    break

                # Handle mode-specific constraints (most are now handled in validation)
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
                        case_state.cumulative_cost + estimated_test_cost
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
                case_state.add_evidence(response)

                # Update costs and state based on action type
                if action.action_type == "test":
                    test_cost = self._estimate_cost(action.content)
                    case_state.cumulative_cost += test_cost
                    case_state.add_test(str(action.content))
                    self.cumulative_cost = case_state.cumulative_cost  # Keep backward compatibility
                    
                    logger.info(f"Tests ordered: {action.content}")
                    logger.info(
                        f"Test cost: ${test_cost:,} | Cumulative cost: ${case_state.cumulative_cost:,}"
                    )
                elif action.action_type == "ask":
                    case_state.add_question(str(action.content))
                    # Questions are part of the same visit until tests are ordered
                    logger.info(f"Questions asked: {action.content}")
                    logger.info(
                        "No additional cost for questions in same visit"
                    )

                # Check budget constraints for budgeted mode
                if (
                    self.mode == "budgeted"
                    and case_state.cumulative_cost >= self.initial_budget
                ):
                    logger.warning(
                        "Budget limit reached. Forcing final diagnosis."
                    )
                    # Use current leading diagnosis
                    final_diagnosis = case_state.get_leading_diagnosis()
                    break

            except Exception as e:
                logger.error(
                    f"Error in diagnostic loop {iteration_count}: {e}"
                )
                # Continue to next iteration or break if critical error
                continue

        else:
            # Max iterations reached without diagnosis
            final_diagnosis = case_state.get_leading_diagnosis()
            if final_diagnosis == "No diagnosis formulated":
                final_diagnosis = "Diagnosis not reached within maximum iterations."
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
            total_cost=case_state.cumulative_cost,
            iterations=iteration_count,
            conversation_history=self.conversation.get_str(),
        )

        logger.info("Diagnostic process completed:")
        logger.info(f"  Final diagnosis: {final_diagnosis}")
        logger.info(f"  Ground truth: {ground_truth_diagnosis}")
        logger.info(f"  Accuracy score: {judgement['score']}/5.0")
        logger.info(f"  Total cost: ${case_state.cumulative_cost:,}")
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

            agg_resp = self._safe_agent_run(aggregator, aggregator_prompt)
            if hasattr(agg_resp, "content"):
                return agg_resp.content.strip()
            return str(agg_resp).strip()

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
                "initial_budget": kwargs.get("budget", 5000),  # Fixed: map budget to initial_budget
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
        
        # Remove 'budget' parameter if present, as it's mapped to 'initial_budget'
        config.pop('budget', None)

        return cls(**config)

    # ------------------------------------------------------------------
    # Helper utilities – throttling & robust JSON parsing
    # ------------------------------------------------------------------

    def _safe_agent_run(
        self,
        agent: "Agent",  # type: ignore – forward reference
        prompt: str,
        retries: int = 3,
        agent_role: AgentRole = None,
    ) -> Any:
        """Safely call `agent.run` while respecting OpenAI rate-limits.

        Features:
        1.  Estimates token usage and provides guidance to agents for self-regulation
        2.  Applies progressive delays to respect rate limits
        3.  Lets agents dynamically adjust their response strategy based on token constraints
        """

        # Get agent role for token calculations
        if agent_role is None:
            agent_role = AgentRole.CONSENSUS  # Default fallback

        # Estimate total tokens in the request
        estimated_input_tokens = self._estimate_tokens(prompt)
        max_output_tokens = self._get_agent_max_tokens(agent_role)
        total_estimated_tokens = estimated_input_tokens + max_output_tokens
        
        # Add dynamic token guidance to the prompt instead of truncating
        token_guidance = self._generate_token_guidance(
            estimated_input_tokens, max_output_tokens, total_estimated_tokens, agent_role
        )
        
        # Prepend token guidance to prompt
        enhanced_prompt = f"{token_guidance}\n\n{prompt}"
        
        logger.debug(f"Agent {agent_role.value}: Input={estimated_input_tokens}, Output={max_output_tokens}, Total={total_estimated_tokens}")

        # Increased base delay for better rate limit compliance
        base_delay = max(self.request_delay, 5.0)  # Minimum 5 seconds between requests

        for attempt in range(retries + 1):
            # Progressive delay: 5s, 15s, 45s, 135s
            current_delay = base_delay * (3 ** attempt) if attempt > 0 else base_delay
            
            logger.info(f"Request attempt {attempt + 1}/{retries + 1}, waiting {current_delay:.1f}s...")
            time.sleep(current_delay)

            try:
                return agent.run(enhanced_prompt)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "ratelimiterror" in err_msg or "429" in str(e):
                    logger.warning(
                        f"Rate-limit encountered (attempt {attempt + 1}/{retries + 1}). "
                        f"Will retry after {base_delay * (3 ** (attempt + 1)):.1f}s..."
                    )
                    continue  # Next retry applies longer delay
                # For non-rate-limit errors, propagate immediately
                raise

        # All retries exhausted
        raise RuntimeError("Maximum retries exceeded for agent.run – aborting call")

    def _robust_parse_action(self, raw_response: str) -> Dict[str, Any]:
        """Extract a JSON *action* object from `raw_response`.

        The function tries multiple strategies and finally returns a default
        *ask* action if no valid JSON can be located.
        """

        import json, re

        # Strip common markdown fences
        if raw_response.strip().startswith("```"):
            segments = raw_response.split("```")
            for seg in segments:
                seg = seg.strip()
                if seg.startswith("{") and seg.endswith("}"):
                    raw_response = seg
                    break

        # 1) Fast path – direct JSON decode
        try:
            data = json.loads(raw_response)
            if isinstance(data, dict) and "action_type" in data:
                return data
        except Exception:
            pass

        # 2) Regex search for the first balanced curly block
        match = re.search(r"\{[\s\S]*?\}", raw_response)
        if match:
            candidate = match.group(0)
            # Remove leading drawing characters (e.g., table borders)
            candidate = "\n".join(line.lstrip("│| ").rstrip("│| ") for line in candidate.splitlines())
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "action_type" in data:
                    return data
            except Exception:
                pass

        logger.error("Failed to parse a valid action JSON. Falling back to default ask action")
        return {
            "action_type": "ask",
            "content": "Could you please clarify the next best step? The previous analysis was inconclusive.",
            "reasoning": "Fallback generated due to JSON parsing failure.",
        }

    def _extract_function_call_output(self, agent_response) -> Dict[str, Any]:
        """Extract structured output from agent function call response.
        
        This method handles the swarms Agent response format when using function calling.
        The response should contain tool calls with the structured data.
        """
        try:
            # Handle different response formats from swarms Agent
            if isinstance(agent_response, dict):
                # Check for tool calls in the response
                if "tool_calls" in agent_response and agent_response["tool_calls"]:
                    tool_call = agent_response["tool_calls"][0]  # Get first tool call
                    if "function" in tool_call and "arguments" in tool_call["function"]:
                        arguments = tool_call["function"]["arguments"]
                        if isinstance(arguments, str):
                            # Parse JSON string arguments
                            import json
                            arguments = json.loads(arguments)
                        return arguments
                
                # Check for direct arguments in response
                if "arguments" in agent_response:
                    arguments = agent_response["arguments"]
                    if isinstance(arguments, str):
                        import json
                        arguments = json.loads(arguments)
                    return arguments
                    
                # Check if response itself has the expected structure
                if all(key in agent_response for key in ["action_type", "content", "reasoning"]):
                    return {
                        "action_type": agent_response["action_type"],
                        "content": agent_response["content"], 
                        "reasoning": agent_response["reasoning"]
                    }
            
            # Handle Agent object response
            elif hasattr(agent_response, "__dict__"):
                # Check for tool_calls attribute
                if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
                    tool_call = agent_response.tool_calls[0]
                    if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                        arguments = tool_call.function.arguments
                        if isinstance(arguments, str):
                            import json
                            arguments = json.loads(arguments)
                        return arguments
                
                # Check for direct function call response
                if hasattr(agent_response, "function_call"):
                    function_call = agent_response.function_call
                    if hasattr(function_call, "arguments"):
                        arguments = function_call.arguments
                        if isinstance(arguments, str):
                            import json
                            arguments = json.loads(arguments)
                        return arguments
                        
                # Try to extract from response content
                if hasattr(agent_response, "content"):
                    content = agent_response.content
                    if isinstance(content, dict) and all(key in content for key in ["action_type", "content", "reasoning"]):
                        return content
            
            # Handle string response (fallback to regex parsing)
            elif isinstance(agent_response, str):
                # Try to parse as JSON first
                try:
                    import json
                    parsed = json.loads(agent_response)
                    if isinstance(parsed, dict) and all(key in parsed for key in ["action_type", "content", "reasoning"]):
                        return parsed
                except:
                    pass
                
                # Fallback to regex extraction
                import re
                action_type_match = re.search(r'"action_type":\s*"(ask|test|diagnose)"', agent_response, re.IGNORECASE)
                content_match = re.search(r'"content":\s*"([^"]+)"', agent_response, re.IGNORECASE | re.DOTALL)
                reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', agent_response, re.IGNORECASE | re.DOTALL)
                
                if action_type_match and content_match and reasoning_match:
                    return {
                        "action_type": action_type_match.group(1).lower(),
                        "content": content_match.group(1).strip(),
                        "reasoning": reasoning_match.group(1).strip()
                    }
            
            logger.warning(f"Could not extract function call output from response type: {type(agent_response)}")
            logger.debug(f"Response content: {str(agent_response)[:500]}...")
            
        except Exception as e:
            logger.error(f"Error extracting function call output: {e}")
            logger.debug(f"Response: {str(agent_response)[:500]}...")
        
        # Final fallback
        return {
            "action_type": "ask",
            "content": "Could you please provide more information to help guide the next diagnostic step?",
            "reasoning": "Fallback action due to function call parsing error."
        }

    def _get_consensus_with_retry(self, consensus_prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """Get consensus decision with function call retry logic."""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt - use original prompt
                    response = self._safe_agent_run(
                        self.agents[AgentRole.CONSENSUS], consensus_prompt, agent_role=AgentRole.CONSENSUS
                    )
                else:
                    # Retry with explicit function call instruction
                    retry_prompt = f"""
{consensus_prompt}

**CRITICAL: RETRY ATTEMPT {attempt}**
Your previous response failed to use the required `make_consensus_decision` function. 
You MUST call the make_consensus_decision function with the appropriate parameters:
- action_type: "ask", "test", or "diagnose"
- content: specific question, test name, or diagnosis
- reasoning: your detailed reasoning

Please try again and ensure you call the function correctly.
"""
                    response = self._safe_agent_run(
                        self.agents[AgentRole.CONSENSUS], retry_prompt, agent_role=AgentRole.CONSENSUS
                    )
                
                logger.debug(f"Consensus attempt {attempt + 1}, response type: {type(response)}")
                
                # Try to extract function call output
                action_dict = self._extract_function_call_output(response)
                
                # Check if we got a valid response (not a fallback)
                if not action_dict.get("reasoning", "").startswith("Fallback action due to function call parsing error"):
                    logger.debug(f"Consensus function call successful on attempt {attempt + 1}")
                    return action_dict
                
                logger.warning(f"Function call failed on attempt {attempt + 1}, will retry")
                
            except Exception as e:
                logger.error(f"Error in consensus attempt {attempt + 1}: {e}")
                
        # Final fallback to JSON parsing if all function call attempts failed
        logger.warning("All function call attempts failed, falling back to JSON parsing")
        try:
            # Use the last response and try JSON parsing
            consensus_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            return self._robust_parse_action(consensus_text)
        except Exception as e:
            logger.error(f"Both function calling and JSON parsing failed: {e}")
            return {
                "action_type": "ask",
                "content": "Could you please provide more information to guide the diagnostic process?",
                "reasoning": f"Final fallback after {max_retries + 1} function call attempts and JSON parsing failure."
            }


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
                    model_name="gemini/gemini-2.5-flash",  # Fixed: Use valid model name
                    max_iterations=5,
                )
            else:
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant,
                    model_name="gemini/gemini-2.5-flash",  # Fixed: Use valid model name
                    max_iterations=5,
                )

            result = orchestrator.run(
                case_info, case_details, ground_truth
            )
            results[variant] = result

        except Exception as e:
            logger.error(f"Error running variant {variant}: {e}")
            results[variant] = None

    return results


# if __name__ == "__main__":
#     # Example case inspired by the paper's Figure 1
#     initial_info = (
#         "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
#         "and bleeding. Symptoms did not abate with antimicrobial therapy."
#     )

#     full_case = """
#     Patient: 29-year-old female.
#     History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
#     No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable. No history of smoking or significant alcohol use.
#     Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
#     Initial Labs: FBC, clotting studies normal.
#     MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
#     Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
#     Biopsy (Immunohistochemistry for Carcinoma): CD31, D2-40, CD34, ERG, GLUT-1, pan-cytokeratin, CD45, CD20, CD3 all negative. Ki-67: 60% nuclear positivity.
#     Biopsy (Immunohistochemistry for Rhabdomyosarcoma): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
#     Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
#     Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
#     """

#     ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

#     # --- Demonstrate Different MAI-DxO Variants ---
#     try:
#         print("\n" + "=" * 80)
#         print(
#             "    MAI DIAGNOSTIC ORCHESTRATOR (MAI-DxO) - SEQUENTIAL DIAGNOSIS BENCHMARK"
#         )
#         print(
#             "                    Implementation based on the NEJM Research Paper"
#         )
#         print("=" * 80)

#         # Test different variants as described in the paper
#         variants_to_test = [
#             (
#                 "no_budget",
#                 "Standard MAI-DxO with no budget constraints",
#             ),
#             ("budgeted", "Budget-constrained MAI-DxO ($3000 limit)"),
#             (
#                 "question_only",
#                 "Question-only variant (no diagnostic tests)",
#             ),
#         ]

#         results = {}

#         for variant_name, description in variants_to_test:
#             print(f"\n{'='*60}")
#             print(f"Testing Variant: {variant_name.upper()}")
#             print(f"Description: {description}")
#             print("=" * 60)

#             # Create the variant
#             if variant_name == "budgeted":
#                 orchestrator = MaiDxOrchestrator.create_variant(
#                     variant_name,
#                     budget=3000,
#                     model_name="gpt-4.1",  # Fixed: Use valid model name
#                     max_iterations=5,
#                 )
#             else:
#                 orchestrator = MaiDxOrchestrator.create_variant(
#                     variant_name,
#                     model_name="gpt-4.1",  # Fixed: Use valid model name
#                     max_iterations=5,
#                 )

#             # Run the diagnostic process
#             result = orchestrator.run(
#                 initial_case_info=initial_info,
#                 full_case_details=full_case,
#                 ground_truth_diagnosis=ground_truth,
#             )

#             results[variant_name] = result

#             # Display results
#             print(f"\n🚀 Final Diagnosis: {result.final_diagnosis}")
#             print(f"🎯 Ground Truth: {result.ground_truth}")
#             print(f"⭐ Accuracy Score: {result.accuracy_score}/5.0")
#             print(f"   Reasoning: {result.accuracy_reasoning}")
#             print(f"💰 Total Cost: ${result.total_cost:,}")
#             print(f"🔄 Iterations: {result.iterations}")
#             print(f"⏱️  Mode: {orchestrator.mode}")

#         # Demonstrate ensemble approach
#         print(f"\n{'='*60}")
#         print("Testing Variant: ENSEMBLE")
#         print(
#             "Description: Multiple independent runs with consensus aggregation"
#         )
#         print("=" * 60)

#         ensemble_orchestrator = MaiDxOrchestrator.create_variant(
#             "ensemble",
#             model_name="gpt-4.1",  # Fixed: Use valid model name
#             max_iterations=3,  # Shorter iterations for ensemble
#         )

#         ensemble_result = ensemble_orchestrator.run_ensemble(
#             initial_case_info=initial_info,
#             full_case_details=full_case,
#             ground_truth_diagnosis=ground_truth,
#             num_runs=2,  # Reduced for demo
#         )

#         results["ensemble"] = ensemble_result

#         print(
#             f"\n🚀 Ensemble Diagnosis: {ensemble_result.final_diagnosis}"
#         )
#         print(f"🎯 Ground Truth: {ensemble_result.ground_truth}")
#         print(
#             f"⭐ Ensemble Score: {ensemble_result.accuracy_score}/5.0"
#         )
#         print(
#             f"💰 Total Ensemble Cost: ${ensemble_result.total_cost:,}"
#         )

#         # --- Summary Comparison ---
#         print(f"\n{'='*80}")
#         print("                           RESULTS SUMMARY")
#         print("=" * 80)
#         print(
#             f"{'Variant':<15} {'Diagnosis Match':<15} {'Score':<8} {'Cost':<12} {'Iterations':<12}"
#         )
#         print("-" * 80)

#         for variant_name, result in results.items():
#             match_status = (
#                 "✓ Match"
#                 if result.accuracy_score >= 4.0
#                 else "✗ No Match"
#             )
#             print(
#                 f"{variant_name:<15} {match_status:<15} {result.accuracy_score:<8.1f} ${result.total_cost:<11,} {result.iterations:<12}"
#             )

#         print(f"\n{'='*80}")
#         print(
#             "Implementation successfully demonstrates the MAI-DxO framework"
#         )
#         print(
#             "as described in 'Sequential Diagnosis with Language Models' paper"
#         )
#         print("=" * 80)

#     except Exception as e:
#         logger.exception(
#             f"An error occurred during the diagnostic session: {e}"
#         )
#         print(f"\n❌ Error occurred: {e}")
#         print("Please check your model configuration and API keys.")
