# cdr3

End-to-end platform for generating and optimizing antibody CDR3 (complementarity-determining region 3) sequences using machine-learning and reinforcement learning techniques.

## Live Project

This repository hosts the codebase. For detailed results, logs or experimental outcomes, refer to the `docs/` or `results/` directories (if present).

## How It’s Made

**Technologies used:**  
Python • PyTorch • fair-esm (or Hugging Face Transformers) • NumPy • pandas • (optional) Scikit-learn • (optional) Hydra / argparse for configuration

**Core pipeline features:**  
- Autoregressive generation of CDR3 sequences based on a transformer policy network  
- Embedding of full heavy-chain sequences (scaffold + CDR3) via ESM‑2 model  
- Reward model for binding score prediction (classifier/regressor)  
- Reinforcement-learning loop (REINFORCE) to drive sequence generation toward higher-scoring designs  
- Utilities for seed-sequence loading, candidate filtering, clustering, and export (FASTA)  

## Lessons Learned

- Integrating large protein language models with downstream custom reward heads  
- Designing a stable policy-gradient loop for sequence generation  
- Handling sequence tokenization, padding/truncation, and decoding in a biological context  
- Balancing exploration vs. exploitation in sequence space  
- Export and downstream filtering of high-scoring candidates for experimental follow-up  

## Features

### Seed & Candidate Management  
- Load seed CDR3 sequences from file  
- Sample minibatches of seeds for generation  
- Deduplicate, filter by length, cluster output sequences  

### Embedding & Scoring  
- Construct full heavy-chain sequence from generated CDR3  
- Featurize full sequence via ESM-2 embedding  
- Score sequences with reward model  

### Generator Policy  
- Transformer-based autoregressive policy network (token embedding + positional embedding + transformer encoder + LM head)  
- Sample sequences via soft sampling or greedy decoding  
- Log-prob tracking for policy loss  

### Training Loop (RL)  
- REINFORCE: update policy with advantage = reward − baseline  
- Entropy regularization to encourage exploration  
- Baseline maintained via momentum average  
- Periodic checkpointing and sequence sampling for inspection  

## Quick Start

```bash
git clone https://github.com/fool0013/cdr3.git
cd cdr3

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
