# PSP408 Interim Poster Content Guide

Use the copy below in the supplied portrait template. The project is at the study-design stage. Dataset counts come from the release paper. The model, cohort experiments, and outcomes are proposed. [Dataset paper] [Template] [Marking guide]

## Header

### Title

**Hierarchical RQ-VAE-Based Clinical Semantic ID Generation for Cross-Disease Transfer in LLM-Driven Discharge Medication Recommendation**

If the title needs two lines, break after “Generation”. Fill in your name, student ID, academy and supervisor as shown in the template.

**Keywords** Medication Semantic IDs · Residual Quantization · Few-Shot Transfer · Chinese Electronic Health Records

关键词可用于提交说明或口头介绍。模板允许省略海报上的关键词。

## Abstract

Can a model trained on one set of disease cohorts predict discharge medications for another cohort using only a few labelled patients? We propose to encode candidate medications as short paths with shared prefixes, then generate medication sets from Chinese electronic health records. We will compare learned paths with randomly reassigned paths on the same decoding tree. Patient-level separation and matched target samples will let us estimate how much the learned mapping contributes to transfer on CDrugRed.

## Introduction

**Background and problem.** CDrugRed reports 5,894 hospital stays from 3,190 patients and a vocabulary of 651 medications. The task is to predict the set of names recorded in the discharge prescription from a Chinese hospital record. Historical prescription agreement is the study endpoint. [Dataset paper] [CHIP overview]

**Motivation and gap.** Learning an independent output for each medication can be difficult when target labels are scarce. Prior work has used discrete representations of medical concepts for medication prediction. This study tests whether organizing the **output medications** into shared prefixes improves transfer to a held-out primary-diagnosis cohort under matched conditions. [TIGER] [MedRQ] [CARing]

**Objectives and questions**

1. Do learned medication prefixes group related drugs and improve set prediction over randomly assigned prefixes?
2. With the same 32 labelled target patients, do learned prefixes increase the benefit of training on source cohorts?

## Methodology

**Medication encoding.** Embed catalogue names and available ingredients with a frozen text model. A residual-quantized variational autoencoder (RQ-VAE) converts each vector into a sequence of codewords. A final suffix uniquely identifies each medication.

**Set generation.** Fine-tune the same Chinese language model on screened clinical text. A catalogue trie restricts decoding to valid medication paths. Decode complete paths to standard drug names and score the resulting set.

**Cohort transfer.** Separate patients before training. Start with other primary-diagnosis cohorts, then adapt on a small labelled sample from the target cohort. Test on distinct patients from that target cohort.

### Figure 1. How the model predicts a medication set

~~~mermaid
flowchart TB
    A["Medication catalogue"] --> B["Text embeddings and RQ-VAE"]
    B --> C["Frozen medication paths"]
    D["Chinese hospital record"] --> E["Matched LLM backbone, trained per arm"]
    C --> F["Catalogue-constrained decoding"]
    E --> F
    F --> G["Predicted discharge medication set"]
~~~

**Caption.** The learned and shuffled mappings use identical paths. Each mapping assigns those paths to medications differently.

## Results and Discussion

### Planned evaluation and current progress

| Comparison | Planned readout |
| --- | --- |
| Learned paths vs. path permutation | Medication-set Jaccard and F1 on an internal patient split |
| Source training plus target adaptation vs. target-only training, for both mappings | Difference in Jaccard gains at 32 target patients, with patient-clustered 95% intervals |
| Shallow text classifier and single-token drug IDs | Practical baselines for set prediction |
| Prefix and label-support analysis | Agreement with independently mapped drug classes, recall for rare or unseen labels, and within-group substitutions |

**Progress.** Benchmark definitions and study design are documented. Data access and cohort feasibility are pending. Once verified, the first experiment is the text classification baseline.

## Conclusion and Future Work

**Expected contribution.** A reusable patient-level evaluation protocol and a controlled test of medication-output hierarchies. The study will estimate when shared prefixes help, have little effect, or reduce accuracy under limited target supervision.

**Expected outcome.** Shared prefixes may help most when target labels are scarce and related medications have appeared in the source cohorts. The experiment will estimate the size and uncertainty of that effect.

**Key risk and next step.** The primary-diagnosis cohorts must contain enough separate patients for adaptation and testing. Check cohort counts and comorbidities after access is approved. If the cohort comparison is infeasible, evaluate the same encodings under low-label conditions in the pooled cohort.

## Selected References

- Li et al. (2025). *CDrugRed*, dataset paper, arXiv v1. [Dataset paper]
- Li et al. (2026). *Overview of CHIP 2025 Shared Task 2*. [CHIP overview]
- Rajput et al. (2023). *Recommender Systems with Generative Retrieval*. NeurIPS. [TIGER]
- Lv et al. (2026). *Towards Efficient and Interpretable Medical Concept Representation via Ontology-driven Residual Vector Quantization*. WWW. [MedRQ]
- Zhang et al. (2026). *Coverage-Aware Reasoning with Medical Tokens for Diagnosis Prediction*. arXiv. [CARing]

## Placement and presentation notes

| Template area | Recommended content |
| --- | --- |
| Full-width header and abstract | Two lines for the title if needed, followed by identity and supervisor information. Use one paragraph for the abstract. |
| Left column, Introduction | Place the task and research questions here. Keep the three dataset numbers together and cite them. |
| Left column, Methodology | Give Figure 1 the largest area in this column. Keep the two input streams visibly distinct and put the explanation of the shuffled control under the diagram. |
| Right column, Results and Discussion | Start with the planned evaluation table and current status. If space permits, add a compact panel showing **source cohorts → 32 target patients → separate target test** alongside **32 target patients → separate target test**. Use the same colour for target test patients in both rows. |
| Right column, Conclusion and Future Work | Put the expected contribution, data access and cohort-size risk, and next milestone here. |
| Footer | Use compact citations. Add acknowledgements if relevant. |

The template is a custom portrait slide, approximately 16.21 × 21.99 inches. Retain its aspect ratio when exporting, and check that the title, questions and figures read at a distance. When results become available, use a paired Jaccard difference with its interval, followed by a learning curve over labelled target patients if space permits. Label the held-out diagnosis cohort and use patient count on the horizontal axis. [Template]

PSP408 specifies a single-page poster, up to five minutes of presentation, and a 10–15 minute Q&A. The marks are distributed across content (40%), design (20%), presentation (20%) and handling questions (20%). The submission instruction in the supplied rubric is 09:00 on Monday of Week 14, Semester 1, Year 2, via Learning Mall. [Marking guide]

### Five-minute speaking plan

| Time | Point to explain |
| --- | --- |
| 0:00–0:50 | Define the discharge label and why a target cohort might have few training patients. |
| 0:50–2:00 | Use the diagram to explain how shared prefixes can transfer information between drugs. |
| 2:00–3:20 | Explain patient separation, the shuffled tree and the four training comparisons. |
| 3:20–4:20 | Identify the Jaccard endpoint, patient-level intervals and prespecified error checks. |
| 4:20–5:00 | State the work completed, the data-access dependency and the next milestone. |

### Q&A preparation

- **Why residual quantization?** It allows multiple drugs to share part of a prediction path. A shuffled mapping tests whether the benefit comes from which drugs share those steps. Atomic IDs and a classifier show the practical cost of adding a hierarchy.
- **What does cross-disease mean here?** It is transfer between primary-diagnosis cohorts in the same hospital. Separate patients and assess a stricter source definition that also excludes identified target comorbidity.
- **What would count as a benefit?** Examine both the extra gain from source training and the final target prediction score. A model can gain more from source training while still scoring below the shuffled mapping.
- **What if the dataset or cohort is unavailable?** Scope depends on approved access and cohort size. The pooled low-label experiment still tests the representation if there are insufficient distinct target patients.
- **Does medication overlap establish clinical quality?** It measures agreement with what was prescribed. Evaluating clinical appropriateness would require additional data and clinical review.

For the full design, see [the research blueprint](MASTER_THESIS_RESEARCH_BLUEPRINT.md).

[Dataset paper]: https://arxiv.org/pdf/2510.21084v1
[CHIP overview]: https://link.springer.com/chapter/10.1007/978-981-95-7299-1_29
[TIGER]: https://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html
[MedRQ]: https://doi.org/10.1145/3774904.3792930
[CARing]: https://arxiv.org/abs/2610.10641
[Template]: <../references/psp408/InterimPoster_Template.ppt>
[Marking guide]: <../references/psp408/PSP408 Interim Poster Presentation Marking Guidelines 26-27.pdf>
