# Open-MAI-Dx-Orchestrator [WIP]

An open source implementation of the paper: "Sequential Diagnosis with Language Models" From Microsoft Built with Swarms Framework.

- [Paper Link](https://arxiv.org/abs/2506.22405)

# Install

```bash
pip3 install mai-dx
```

## Usage

...

---

## Architecture

### Virtual Panel Roles

The virtual panel consists of five specialized roles:

- **Dr. Hypothesis** – Maintains a probability-ranked differential diagnosis with the top three most likely conditions, updating probabilities in a Bayesian manner after each new finding.

- **Dr. Test-Chooser** – Selects up to three diagnostic tests per round that maximally discriminate between leading hypotheses.

- **Dr. Challenger** – Acts as devil's advocate by identifying potential anchoring bias, highlighting contradictory evidence, and proposing tests that could falsify the current leading diagnosis.

- **Dr. Stewardship** – Enforces cost-conscious care by advocating for cheaper alternatives when diagnostically equivalent and vetoing low-yield expensive tests.

- **Dr. Checklist** – Performs silent quality control to ensure the model generates valid test names and maintains internal consistency across the panel's reasoning.

### Decision Process

After internal deliberation, the panel reaches consensus on one of three actions:
- Asking questions
- Ordering tests  
- Committing to a diagnosis (if certainty exceeds threshold)

Before tests are ordered, an optional budget tracker can be invoked to estimate both the cumulative medical costs so far and the cost of each test in the order.

### MAI-DxO Variants

We evaluate five variants of MAI-DxO to explore different points on the accuracy-cost frontier (from most cost conscious to least):

- **Instant Answer** – Diagnosis based solely on initial vignette (as in Figure 3), without any follow-up questions or tests.

- **Question Only** – The panel can ask questions, but cannot order diagnostic tests. The cost is simply the cost of a single physician visit.

- **Budgeted** – The panel is augmented with a budgeting system that tracks cumulative costs (a separately orchestrated language model call) towards a max budget and allows the panel to cancel tests after seeing their estimated cost.

- **No Budget** – Full panel with no explicit cost tracking or budget limitations.

- **Ensemble** – Simulates multiple doctor panels working in parallel, with an additional panel to provide a final diagnosis. This is implemented as multiple independent No Budget runs with a final aggregation step to select the best diagnosis. Costs are computed as the sum of the costs of all tests ordered by each of the runs, accounting for duplicates.

### Technical Implementation

MAI-DxO was primarily developed and optimized using GPT-4.1, but is designed to be model-agnostic. All MAI-DxO variants used the same underlying orchestration structure, with capabilities selectively enabled or disabled for variants.

## Citation

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
```



# License
MIT
