# ReLATE  
**Relation-Level Automated Template Evaluation for NLI Explanations**

---

## Overview

This project implements an evaluation system and automatic generation pipeline for NLI explanations using the **e-SNLI dataset** (Camburu et al., 2018).

It builds on the work of Milanese et al. (2025) and focuses on one specific relation: entailment in the form of subset relations (`⊑`). Human-provided explanations are transformed into structured subset relations by matching the words surrounding subset expressions ("type of", "kind of", "form of") with annotators’ highlighted tokens in the premise and hypothesis.

The full pipeline consists of:

1. Preprocessing the original e-SNLI dataset  
2. Extracting structured subset relations from human explanations  
3. Generating LLM explanations via Gemini  
4. Parsing model outputs into structured format  
5. Computing relation-level evaluation metrics  

---

## Repository Structure

Below is an overview of the main files in the repository:


- `esnli_dev.csv`
  The development split of the e-SNLI dataset.

- `esnli_test.csv`
  The test split of the e-SNLI dataset.

- `entailment_probs_&.csv`
  Contains all problems from the developement split where all three annotators used "type of" "form of" or "kind of".

- `entailment_probs_2.csv`
  Contains all problems from the developement split where at least two annotators used "type of" "form of" or "kind of".

- `entailment_probs_or.csv`
  Contains all problems from the developement split where any of the three annotators used "type of" "form of" or "kind of"

- `entailment_probs_test_or.csv`
  Contains all problems from the test split where any of the three annotators used "type of" "form of" or "kind of"

- `merged_entailment.csv`
  Contains all problems from the developement and test splits where any of the three annotators used "type of" "form of" or "kind of".

- `60_annotated_problems.csv`  
  Contains the 60 manually annotated problems used for template validation.


- `annotator_agreement.py`  
  Generates plots used in the paper.  
  If `Get_manual_evaluation_problems(True, True)` is used, it prints full details of the 60 manually annotated problems.

- `auto_evaluation.py`  
  End-to-end evaluation pipeline.  
  Can:
  - Generate new LLM predictions via the Gemini API  
  - Evaluate existing JSON outputs without regenerating  

  Computes:
  - Strict and loose relation-level precision, recall, F1  
  - Length-based scores  
  - Entailment rates  
  - Diagnostic statistics  

- `evaluation.py`  
  Contains all evaluation and scoring functions.

- `generation.py`  
  Handles LLM generation and output parsing.

- `processing_EA.py`  
  Preprocesses e-SNLI and extracts the subset used in the experiments.

- `prompts_construction.py`  
  Builds prompts by combining the fixed prompt with generated problems.

- `explore_esnli_data.py`  
  Used for dataset inspection and debugging.

- `test_max_problems.py`  
  Prints problems where no answer template could be extracted and reports exclusion statistics.

- `fixed.txt`  
  Fixed part of the LLM prompt.

- `LLM_file.txt`  
  Contains formatted problems fed to the LLM.

- `id_map.json`  
  Maps shortened IDs (`q0001`, etc.) to original e-SNLI pair IDs.

- `gold.json`  
  Gold annotated answer templates.

- `final_LLM_auto_responses_*.json`  
  Parsed LLM outputs for Gemini models.

- `requirements.txt`  
  Python dependencies.

---

## Requirements

- Python 3.11  
- Conda (recommended)  
- Gemini API key (only required for generation)

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create Environment

```bash
conda create -n NLI python=3.11
conda activate NLI
pip install -r requirements.txt
```

## Gemini API Setup (Only for Generation)

To generate new model outputs, you need a Gemini API key.

1. Generate a free API key at:  
   https://aistudio.google.com/api-keys

2. Set the environment variable:

### macOS / Linux

```bash
export GEMINI_API_KEY="your-api-key"
```

### Windows

```bash
setx GEMINI_API_KEY "your-api-key"
```

> Note: You must repeat this command each time you open a new terminal session.


## Running the Evaluation

Open `auto_evaluation.py` and set:

```python
flag = True
```

- `flag = True` → Evaluate existing JSON files (reproduces paper results)
- `flag = False` → Generate new LLM explanations and evaluate them


### Run the Script

#### macOS / Linux

```bash
python3 auto_evaluation.py
```

#### Windows

```bash
python auto_evaluation.py
```

---

## Output Metrics

The script prints:

- Percentage of entailment predictions  
- Strict precision / recall / F1  
- Loose precision / recall / F1  
- Length-based metrics  
- Average explanation lengths  
- Diagnostic statistics (missing relations, contradictions, etc.)

---

## Reproducibility

To reproduce the results reported in the paper:

```python
flag = True
```

No API key is required for this mode.

---

## Method Summary

The evaluation operates at the **relation level**, not the problem level.

Two evaluation metrics are defined:

### Strict Metric
A predicted relation is correct if:
- All gold tokens are present  
- At most two extra tokens are included  

### Loose Metric
A predicted relation is correct if:
- At least one token overlaps with a gold relation  

Precision, recall, and F1 are computed under both settings.

---

## License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.
