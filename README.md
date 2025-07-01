# MAI Diagnostic Orchestrator (MAI-DxO)

> **AI-powered diagnostic system that simulates a virtual panel of physician-agents for medical diagnosis**

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.22405-red.svg)](https://arxiv.org/abs/2506.22405)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

An open-source implementation of Microsoft Research's "Sequential Diagnosis with Language Models" paper, built with the Swarms AI framework.

## 🚀 Quick Start

```bash
git clone https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator.git
cd Open-MAI-Dx-Orchestrator
pip install -r requirements.txt
```

```python
from mai_dx import MaiDxOrchestrator

# Create orchestrator
orchestrator = MaiDxOrchestrator()

# Run diagnosis
result = orchestrator.run(
    initial_case_info="29-year-old woman with sore throat and peritonsillar swelling...",
    full_case_details="Patient: 29-year-old female. History: Onset of sore throat...",
    ground_truth_diagnosis="Embryonal rhabdomyosarcoma of the pharynx"
)

print(f"Diagnosis: {result.final_diagnosis}")
print(f"Accuracy: {result.accuracy_score}/5.0")
print(f"Cost: ${result.total_cost:,}")
```

## ✨ Key Features

- **8 AI Physician Agents**: Specialized roles for comprehensive diagnosis
- **5 Operational Modes**: instant, question-only, budgeted, no-budget, ensemble
- **Cost Tracking**: Real-time budget monitoring with 25+ medical test costs
- **Clinical Evaluation**: 5-point accuracy scoring with detailed feedback
- **Model Agnostic**: Works with GPT, Gemini, Claude, and other LLMs

## 🏥 Virtual Physician Panel

- **🧠 Dr. Hypothesis**: Maintains differential diagnosis with probabilities
- **🔬 Dr. Test-Chooser**: Selects optimal diagnostic tests
- **🤔 Dr. Challenger**: Prevents cognitive biases and diagnostic errors
- **💰 Dr. Stewardship**: Ensures cost-effective care decisions
- **✅ Dr. Checklist**: Quality control and consistency checks
- **🤝 Consensus Coordinator**: Synthesizes panel decisions
- **🔑 Gatekeeper**: Clinical information oracle
- **⚖️ Judge**: Evaluates diagnostic accuracy

## 📋 Usage Modes

```python
# Instant diagnosis (emergency triage)
orchestrator = MaiDxOrchestrator.create_variant("instant")

# Budget-constrained diagnosis
orchestrator = MaiDxOrchestrator.create_variant("budgeted", budget=3000)

# Question-only mode (telemedicine)
orchestrator = MaiDxOrchestrator.create_variant("question_only")

# Full diagnostic capability
orchestrator = MaiDxOrchestrator.create_variant("no_budget")

# Ensemble approach (multiple panels)
result = orchestrator.run_ensemble(case_info, case_details, ground_truth, num_runs=3)
```

## 🛠 Configuration

```python
orchestrator = MaiDxOrchestrator(
    model_name="gemini/gemini-2.5-flash",  # or "gpt-4", "claude-3-5-sonnet"
    max_iterations=10,
    initial_budget=10000,
    mode="no_budget"
)
```

## 📚 Documentation

- **[Complete Documentation](docs.md)** - Detailed API reference and examples
- **[Example Usage](example.py)** - Ready-to-run examples
- **[Original Paper](https://arxiv.org/abs/2506.22405)** - Microsoft Research paper

## 🎯 Example Results

```
=== MAI-DxO Diagnostic Results ===
Variant: no_budget
Final Diagnosis: Embryonal rhabdomyosarcoma of the pharynx
Ground Truth: Embryonal rhabdomyosarcoma of the pharynx
Accuracy Score: 5.0/5.0
Total Cost: $4,650
Iterations: 4
```

## 🤝 Contributing

We welcome contributions! Please check our issues and submit pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@misc{nori2025sequentialdiagnosislanguagemodels,
      title={Sequential Diagnosis with Language Models}, 
      author={Harsha Nori and others},
      year={2025},
      eprint={2506.22405},
      archivePrefix={arXiv}
}
```
