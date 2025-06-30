# Open-MAI-Dx-Orchestrator

> **An open-source implementation of the "Sequential Diagnosis with Language Models" paper by Microsoft Research, built with the Swarms AI framework.**

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.22405-red.svg)](https://arxiv.org/abs/2506.22405)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

MAI-DxO (MAI Diagnostic Orchestrator) is a sophisticated AI-powered diagnostic system that simulates a virtual panel of physician-agents to perform iterative medical diagnosis with cost-effectiveness optimization. This implementation faithfully reproduces the methodology described in the Microsoft Research paper while providing additional features and flexibility.

## 🚀 Quick Start

```bash
# Install the package
pip install mai-dx

# Or install from source
git clone https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator.git
cd Open-MAI-Dx-Orchestrator
pip install -e .
```

---- 

## Enivronement Configuration
Configure your api keys and environment variables like the following. 

```txt
WORKSPACE_DIR="" # for the swarms library
OPENAI_API_KEY="" # Your model api key
GEMINI_API_KEY="" # your gemini api key
ANTHROPIC_API_KEY=""
```

----

## Example

```python
from mai_dx import MaiDxOrchestrator

# Create orchestrator
orchestrator = MaiDxOrchestrator(model_name="gemini/gemini-2.5-flash")

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

--- 

## Documentation

Learn more about this repository [with the docs](DOCS.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this implementation in your research, please cite both the original paper and this implementation:

```bibtex
@misc{nori2025sequentialdiagnosislanguagemodels,
    title={Sequential Diagnosis with Language Models}, 
    author={Harsha Nori and Mayank Daswani and Christopher Kelly and Scott Lundberg and Marco Tulio Ribeiro and Marc Wilson and Xiaoxuan Liu and Viknesh Sounderajah and Jonathan Carlson and Matthew P Lungren and Bay Gross and Peter Hames and Mustafa Suleyman and Dominic King and Eric Horvitz},
    year={2025},
    eprint={2506.22405},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2506.22405}, 
}

@software{mai_dx_orchestrator,
    title={Open-MAI-Dx-Orchestrator: An Open Source Implementation of Sequential Diagnosis with Language Models},
    author={The-Swarm-Corporation},
    year={2025},
    url={https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator.git}
}
```

## 🔗 Related Work

- [Original Paper](https://arxiv.org/abs/2506.22405) - Sequential Diagnosis with Language Models
- [Swarms Framework](https://github.com/kyegomez/swarms) - Multi-agent AI orchestration
- [Microsoft Research](https://www.microsoft.com/en-us/research/) - Original research institution

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator/discussions)
- **Documentation**: [Full Documentation](https://docs.swarms.world)

---

<p align="center">
  <strong>Built with Swarms for advancing AI-powered medical diagnosis</strong>
</p>
