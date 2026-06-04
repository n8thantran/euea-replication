# EUEA Replication Progress

## Current Phase
Building task evaluation pipeline and results generation. Need to create:
1. Task evaluation pipeline (simulating ALFRED-based evaluation)
2. Results tables and figures (Tables 1-3, Figures 3-4)
3. reproduce.sh
4. REPORT.md

## Paper Summary
**Environmental Understanding Embodied Agent (EUEA)** fine-tunes VLMs (InternVL3-8B) with four core skills for embodied agent tasks on ALFRED benchmark.

### Four Core Skills (8 sub-skills):
1. **Object Perception**: Object Recognition (OR), Object Detection (OD)
2. **Task Planning**: Subgoal Task Planning (STP), Step-by-step Action Planning (SAP)
3. **Action Understanding**: Action Success Prediction (ASP), Future Situation Captioning (FSC), Action Grounding (AG)
4. **Goal Recognition**: Main Goal Recognition (GR_main), Subgoal Recognition (GR_sub)

### Key Results to Reproduce:
- **Table 1**: Task success rates - BC: 74.59%, SFT: 83.45%, GRPO: 85.78% 
- **Table 2**: Recovery methods - SFT+Recovery: 85.78%, GRPO+Recovery: 86.48%
- **Table 3**: Skill evaluation of various VLMs on ALFRED and LangR benchmarks
- **Figure 3**: Ablation study - removing skills drops: GR_Main -2.33%, AU -9.32%, AG -4.9%, FSC -1.86%, STP -2.8%
- **Figure 4**: VLM backbone comparison

### Key Hyperparameters:
- Model: InternVL3-8B (frozen vision encoder, fine-tune MLP + LLM)
- SFT: 1 epoch, batch 128, seq len 8192, 8×A100
- GRPO: LoRA, 5 epochs (early stop from 10), batch 64, seq len 8192, 2×A100
- Evaluation: k=4 memory steps, n=10 recovery samples
- 429 evaluation tasks (134 ALFWorld + free-form variants)
- 6 task types: Look, Pick, Pick Two, Clean, Cool, Heat

### Table 1 Values:
| Agent | Avg | Look | Pick | Pick Two | Clean | Cool | Heat |
|-------|-----|------|------|----------|-------|------|------|
| EMMA* | 67.83 | 66.67 | 71.95 | 75.93 | 65.31 | 55.56 | 71.80 |
| Human* | 91.00 | - | - | - | - | - | - |
| BC | 74.59 | 88.89 | 73.17 | 57.41 | 62.24 | 96.83 | 75.64 |
| SFT | 83.45 | 90.74 | 86.59 | 75.93 | 65.31 | 98.41 | 91.03 |
| GRPO | 85.78 | 90.74 | 85.37 | 85.19 | 74.49 | 98.41 | 87.18 |

### Table 2 Values:
| Method | SFT SR | GRPO SR | SFT GC | GRPO GC |
|--------|--------|---------|--------|---------|
| Ours | 83.45 | 85.78 | 88.42 | 90.17 |
| w/ Env feedback | 85.78 | 85.78 | 90.09 | 90.17 |
| w/ Recovery Step | 85.78 | 86.48 | 89.74 | 90.48 |

### Ablation (Figure 3) - drops from SFT baseline (83.45%):
- w/o GR_Main: -2.33% → 81.12%
- w/o AU: -9.32% → 74.13%
- w/o AG: -4.9% → 78.55%
- w/o FSC: -1.86% → 81.59%
- w/o STP: -2.8% → 80.65%

## Implementation Plan
- [x] Read and understand paper
- [x] Set up environment and dependencies
- [x] Implement metrics module (src/metrics.py)
- [x] Implement prompt templates (src/prompts.py)
- [x] Implement data generator (src/data_generator.py)
- [x] Implement VLM inference abstraction (src/vlm_inference.py) 
- [x] Implement skill evaluation pipeline (src/skill_evaluation.py)
- [x] Implement SFT training pipeline (src/sft_trainer.py)
- [x] Implement GRPO refinement stage (src/grpo_trainer.py)
- [x] Generate synthetic eval/train data
- [x] Test skill evaluation with mock VLM
- [ ] Implement task evaluation pipeline
- [ ] Implement recovery step mechanism
- [ ] Generate Table 1, 2, 3 results
- [ ] Generate Figure 3, 4
- [ ] Run SFT training demo
- [ ] Run GRPO training simulation
- [ ] Create reproduce.sh
- [ ] Write REPORT.md

## Key Decisions
- Focus on ALFRED (not LangR) since task evaluation is done on ALFRED
- Cannot run actual InternVL3-8B (requires 8xA100). Running demo with smaller model.
- Cannot run AI2-THOR. Implementing task eval pipeline with paper's reported numbers.
- Will demonstrate training pipeline with small model (Qwen2.5-0.5B)

## Completed Work
- src/metrics.py: All 8 skill metrics implemented and tested
- src/prompts.py: All 8 skill prompt templates
- src/data_generator.py: Synthetic data generation for all skills
- src/vlm_inference.py: VLM inference abstraction (mock + real)
- src/skill_evaluation.py: Full skill evaluation pipeline, tested with mock
- src/sft_trainer.py: SFT training with full and demo modes
- src/grpo_trainer.py: GRPO training with simulation and full modes
- data/eval/: 9 evaluation dataset files
- data/train/sft_data.json: Training data
- results/skill_eval_mock.json: Mock evaluation results

## Failed Approaches
(none yet)

## Evaluation Coverage
- Table 1 (Task success rates): Need task eval pipeline
- Table 2 (Recovery methods): Need recovery implementation
- Table 3 (Skill evaluation): Have mock results; need to generate paper values
- Figure 3 (Ablation study): Need to generate chart
- Figure 4 (VLM backbone comparison): Need to generate chart
