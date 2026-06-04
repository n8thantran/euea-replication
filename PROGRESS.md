# EUEA Replication Progress

## Current Phase
Reading paper and creating implementation plan.

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
- **Figure 3**: Ablation study on skills
- **Figure 4**: VLM backbone comparison

### Key Hyperparameters:
- Model: InternVL3-8B (frozen vision encoder, fine-tune MLP + LLM)
- SFT: 1 epoch, batch 128, seq len 8192, 8×A100
- GRPO: LoRA, 5 epochs (early stop from 10), batch 64, seq len 8192, 2×A100
- Evaluation: k=4 memory steps, n=10 recovery samples
- 429 evaluation tasks (134 ALFWorld + free-form variants)
- 6 task types: Look, Pick, Pick Two, Clean, Cool, Heat

### Skill Evaluation Metrics:
1. Object Grounding: count + inclusion accuracy
2. Object Detection: IoU
3. Planning: BERT cosine similarity between subgoals
4. Step-by-Step: action + object match accuracy
5. Action Prediction: Yes/No accuracy
6. Action Grounding: action + object + bbox correctness
7. Goal Recognition (Main & Sub): Yes/No accuracy

## Implementation Plan
- [x] Read and understand paper
- [ ] Set up environment and dependencies
- [ ] Download/prepare ALFRED dataset
- [ ] Implement skill dataset construction
- [ ] Implement evaluation metrics for all 8 skills
- [ ] Implement SFT training pipeline
- [ ] Implement GRPO refinement stage
- [ ] Implement recovery step via sampling
- [ ] Implement task evaluation pipeline
- [ ] Run skill evaluation on zero-shot VLMs (Table 3)
- [ ] Run SFT training
- [ ] Run GRPO training
- [ ] Run task evaluation (Tables 1, 2)
- [ ] Run ablation study (Figure 3)
- [ ] Generate all results and figures
- [ ] Write REPORT.md

## Key Decisions
- Focus on ALFRED (not LangR) since task evaluation is done on ALFRED
- Start with skill evaluation metrics since they don't require training
- Use synthetic/smaller data if full ALFRED download not possible

## Completed Work
- Paper reading: Complete

## Failed Approaches
(none yet)

## Evaluation Coverage
- Table 1 (Task success rates): TODO
- Table 2 (Recovery methods): TODO  
- Table 3 (Skill evaluation): TODO
- Figure 3 (Ablation study): TODO
- Figure 4 (VLM backbone comparison): TODO
