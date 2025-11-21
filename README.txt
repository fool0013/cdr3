# ABYSS – Antibody Binding Yield through Sequence Searching

ABYSS is a computational antibody design platform that generates and optimizes antibody CDR3 (Complementarity-Determining Region 3) sequences for improved antigen binding.

The system combines:
- ESM-2 protein language models
- A learned scoring model for antigen–CDR3 pairs
- An iterative beam search and mutation pipeline
- Diversity clustering and candidate selection

Live app: https://cdr3byss.vercel.app/

---

## Project Description

CDR3 is the most variable and functionally critical region of an antibody, located at the center of the antigen-binding site. It directly contacts the target antigen and heavily influences binding affinity and specificity.

Traditional antibody discovery relies on immunization or display technologies, which are slow and expensive. ABYSS approaches the problem computationally:

1. Takes an antigen sequence as input.
2. Generates CDR3 sequences using an iterative, search-based pipeline.
3. Embeds antigen–CDR3 pairs using ESM-2.
4. Scores candidates with a trained scoring model that predicts binding likelihood.
5. Filters, clusters, and returns a diverse set of top candidates in CSV and FASTA format.

ABYSS is designed as:
- A **web app** for interactive configuration, optimization, and inspection.
- A **CLI / Python pipeline** for batch runs and offline experimentation.

---

## How to Use the App (Web Version)

Live URL: https://cdr3byss.vercel.app/

The app is organized into several tabs visible in the navigation bar:

### 1. Overview

The **Overview** tab explains:
- What CDR3 is and why it matters
- The challenges of antibody design
- The ABYSS approach: ESM-2 embedding, scoring model, iterative beam search, and diversity clustering
- The high-level pipeline: **Generate → Filter → Cluster**

Use this tab as the conceptual starting point.

### 2. Config – Set Parameters

Go to the **Config** tab when you are ready to run an experiment.

Here, you configure:

- **Antigen sequence**  
  Paste the amino-acid sequence of your target antigen.

- **CDR3 length range**  
  Minimum and maximum CDR3 length to explore (e.g., 10–18).

- **Beam search / candidate settings**  
  - Number of initial seeds
  - Beam width (how many top sequences to keep each step)
  - Number of optimization steps / iterations
  - Mutation rate or number of mutations per step

- **Diversity and filtering options**  
  - Minimum Hamming distance or similarity threshold for diversity
  - Score thresholds to filter low-quality sequences

When you are finished, save or apply the configuration. These settings are used by the **Optimize** tab.

### 3. Optimize – Run ABYSS

Go to the **Optimize** tab to run the main pipeline.

Typical usage:

1. Confirm the configuration (antigen, length range, and search parameters).
2. Click the **Run** or **Start Optimization** button.
3. The app will:
   - Generate candidate CDR3 sequences
   - Embed antigen–CDR3 pairs with ESM-2
   - Score each candidate using the scoring model
   - Apply beam search and mutation over multiple steps
   - Filter and cluster sequences to maintain diversity

A progress indicator shows the current step. When complete, a summary of how many sequences passed each stage is displayed. Detailed results are available under **Results**.

### 4. Results – Inspect and Download Sequences

Open the **Results** tab after a run completes.

Here you can:

- View a table of generated CDR3 candidates, including:
  - CDR3 sequence
  - Predicted binding score
  - Sequence length
  - Cluster / diversity group
- Sort by score or length
- Filter by score threshold, length range, or cluster ID
- Select subsets of sequences interactively

Export options typically include:

- **Download CSV** – scores, sequences, and metadata
- **Download FASTA** – sequences in FASTA format for downstream tools

Use this tab to identify the top few candidates for experimental validation or further analysis.

### 5. Train – Custom Scoring Model

The **Train** tab lets you train a new scoring model on custom antigen–CDR3 pairs.

Typical workflow:

1. Prepare a dataset file (CSV) containing:
   - Antigen sequence
   - CDR3 sequence
   - Label or binding score (e.g., 0/1 or continuous)
2. Upload the dataset.
3. Configure training parameters (epochs, batch size, learning rate).
4. Start training.

The app will compute ESM-2 embeddings for your data and train a neural network to predict binding probability. The resulting model can replace or augment the default scoring model used in **Optimize**.

### 6. Evaluate – Model Performance

The **Evaluate** tab is used to assess the trained model.

It can display:

- ROC curve and AUC
- Precision/recall metrics
- Calibration or score distribution plots
- Example positives/negatives

This helps verify that the model is learning meaningful patterns before you rely on it for large-scale sequence generation.

---

## How to Launch the Code Locally

The repository supports both:

- A **Python CLI / backend** for running the generation pipeline.
- A **Next.js frontend** for the web UI.

### 1. Clone the repo
git clone https://github.com/fool0013/cdr3.git
cd cdr3

### 2. create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

### 3. install dependencies
pip install -r requirements.txt

### 4. finally, start the cli.
python cdr3_cli.py

