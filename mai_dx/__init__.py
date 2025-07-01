"""
MAI Diagnostic Orchestrator (MAI-DxO)

An AI-powered diagnostic system that simulates a virtual panel of physician-agents
for medical diagnosis. Based on Microsoft Research's "Sequential Diagnosis with 
Language Models" paper.

Key Components:
- MaiDxOrchestrator: Main orchestrator class
- DiagnosisResult: Result container with diagnosis, accuracy, and cost
- Action: Action model for panel decisions
- AgentRole: Enumeration of physician agent roles
"""

from typing import Any, Optional

from .main import (
    MaiDxOrchestrator,
    DiagnosisResult,
    Action,
    AgentRole,
    run_mai_dxo_demo
)

__version__ = "1.0.0"
__author__ = "The Swarm Corporation"
__email__ = "kye@apac.ai"
__description__ = "AI-powered diagnostic system with virtual physician panels"

# Main exports
__all__ = [
    "MaiDxOrchestrator",
    "DiagnosisResult", 
    "Action",
    "AgentRole",
    "run_mai_dxo_demo"
]

# Convenience imports for common usage patterns
def create_orchestrator(
    model_name: str = "gemini/gemini-2.5-flash",
    mode: str = "no_budget",
    **kwargs: Any
) -> MaiDxOrchestrator:
    """
    Convenience function to create a MAI-DxO orchestrator.
    
    Args:
        model_name: Language model to use
        mode: Operational mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MaiDxOrchestrator instance
    """
    if mode in ["instant", "question_only", "budgeted", "no_budget", "ensemble"]:
        return MaiDxOrchestrator.create_variant(mode, model_name=model_name, **kwargs)
    else:
        return MaiDxOrchestrator(model_name=model_name, mode=mode, **kwargs)


def quick_diagnosis(
    case_info: str,
    case_details: str,
    ground_truth: Optional[str] = None,
    model_name: str = "gemini/gemini-2.5-flash"
) -> DiagnosisResult:
    """
    Convenience function for quick diagnosis without configuration.
    
    Args:
        case_info: Initial case presentation
        case_details: Complete case information
        ground_truth: Correct diagnosis (optional)
        model_name: Model to use
        
    Returns:
        DiagnosisResult with diagnosis and evaluation
    """
    orchestrator = MaiDxOrchestrator(model_name=model_name, max_iterations=5)
    return orchestrator.run(
        initial_case_info=case_info,
        full_case_details=case_details,
        ground_truth_diagnosis=ground_truth or "Unknown"
    )
