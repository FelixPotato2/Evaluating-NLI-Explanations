# Structured-e-SNLI
**Structered explanations for SNLI** 

Explanations in Natural Language Inference (NLI) remain predominantly unstructured, making systematic comparison and interpretability challenging.
The e-SNLI (Camburu et al.,2018) dataset extends SNLI by providing human-written explanations, but their variability in phrasing and granularity hinders structured analysis. 
In this project, we introduce a rule-based approach to extract Structured Explanations, transforming free-text justifications into formal representations consisting of operators and arguments. Our method leverages syntactic parsing and pattern matching to identify recurring linguistic structures.
We evaluate our approach quantitatively by measuring dataset coverage, which refers to the proportion of explanations that could be successfully structured using predefined rules, and qualitatively through manual alignment with human reasoning. The results indicate that our method effectively captures structured reasoning for a significant portion of explanations, with higher coverage in contradiction cases than in entailment or neutral cases. 
Although structured representations improve interpretability, limitations related to annotation inconsistencies and syntactic variability remain.

**Preprocessing**

In the preprocessing.py we preprocessed the e-SNLI dataset according to our needs of computation.

**Patterns Matching**

In the folder patterns of this repository you can find entailment.py, contradiction.py and neutral.py. These three files are key for our pattern matching computation. For each of the label - entailment, contradiction and neutral - we defined patterns based on the most frequent phrases and words we found in the explanation. Once these patterns are detected, we create a structured explanation out of this detection. In abstract.py you can find utilities functions that we use in this phase.

**Quantitative analysis**

We conduced a quantitative analysis by computing the coverage - proportion of explanations that could be successfully structured using predefined rules - for each label. You can find this analysis in eval.ipynb.

**Qualitative analysis**

For the qualitative analysis we computed both accuracy and recall considering manual alignment with human reasoning (data - assigned_samples_training). The qualitative analysis is in quality_test.ipynb
