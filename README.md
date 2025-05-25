
# Situat3DChange

**Situat3DChange** is a 3D visual-language benchmark designed to assess multimodal large language models (MLLMs) on real-world situation understanding tasks, including change detection, situation description, rearrangement planning, and question answering, all based on complex spatial, object-centric scene changes.

- 📂 Dataset on Hugging Face: [lrp123/Situat3DChange](https://huggingface.co/datasets/lrp123/Situat3DChange)
- 🤖 Baseline model: **SCReasoner**
- 📊 Evaluation tools: for both traditional NLP metrics and GPT-based evaluation

---

## 📦 Installation

We recommend setting up the environment by following the steps in [embodied-generalist](https://github.com/embodied-generalist/embodied-generalist), as SCReasoner builds on similar infrastructure.

Clone the repo:
```bash
git clone https://github.com/RuipingL/Situat3DChange.git
cd Situat3DChange
```

---

## 🚀 SCReasoner Setup & Training

1. **Download Checkpoints**

Download `checkpoints.zip` from the [Hugging Face dataset page](https://huggingface.co/datasets/lrp123/Situat3DChange/blob/main/checkpoints.zip), and extract it into:
```
Situat3DChange/SCReasoner/
```

2. **Launch Training**

Use the following command to train SCReasoner with SLURM and Submitit:

```bash
python launch.py \
  --mode submitit \
  --config configs/default.yaml \
  --name default \
  --time 48 \
  --num_nodes 1 \
  --partition accelerated \
  --gpu_per_node 4 \
  --mem_per_gpu 100 \
  --port 2050
```

---

## 🧪 Evaluation

### 1. QA Task

Run:
```bash
python eval_qa/eval.py
```

### 2. Longform Tasks

For **traditional metrics** (BLEU-4, ROUGE, CIDEr, METEOR, BERTScore):
```bash
python eval_longform/eval.py
```

For **GPT-based evaluation**:
```bash
python eval_longform/eval_gpt.py
```

---

## 📁 Results

Results for **SCReasoner** including GPT scores are stored in:
```
results/SCReasoner/
```

---

## 📫 Citation

If you use this project or dataset, please cite us (citation coming soon).
