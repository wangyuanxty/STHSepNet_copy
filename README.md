# STH-SepNet

[KDD 2025] **Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs**

[Paper](https://arxiv.org/abs/2505.19620) | [Original Code](https://github.com/jiawenchen10/STHSepNet)

**Abstract**: Spatio-temporal prediction is a pivotal task with broad applications in traffic management, climate monitoring, and energy scheduling. STH-SepNet decouples temporal and spatial modeling: a lightweight LLM captures low-rank temporal dynamics, an adaptive hypergraph neural network models higher-order spatial interactions, and a gating mechanism fuses both representations.

---

## Open Problem Solutions

This repository extends STH-SepNet with two solutions addressing the open problems identified in the paper (Section 5, "Limitations and future work"):

### Solution 1: Cluster Pooling

**Problem**: `x_enc.mean(2)` pools all N sensor curves into a single global mean, erasing spatial diversity. A crash at sensor A (65→8) gets averaged with 357 normal sensors — the LLM sees nothing.

**Solution**: Replace global mean with learnable soft clustering. Node embeddings (shared with the hypergraph constructor) drive C cluster centers. The LLM sees C differentiated curves with spatial identity, not one blurred average.

```
x_enc.mean(2)              →  N→1→LLM→1→N  (original)
soft_cluster_pool(x_enc)   →  N→C→LLM→C→N  (ours, C≪N)
```

### Solution 2: Neural Hypergraph Generator

**Problem**: The adaptive hypergraph uses sklearn `NearestNeighbors` (KNN) to build hyperedges — non-differentiable and requires CPU-GPU data transfers.

**Solution**: Replace KNN with a differentiable MLP scoring network + Gumbel-Softmax Top-K. All operations are GPU-native matrix multiplications, and the hypergraph structure receives gradient feedback from the prediction loss.

```
NearestNeighbors.kneighbors()   →  KNN, non-diff, CPU (original)
MLP_scorer + GumbelSoftmax     →  neural, diff, GPU (ours)
```

### Quick Start

```bash
# Baseline (original STH-SepNet)
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295

# Solution 1: Cluster Pooling
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_cluster_pooling --num_clusters 8

# Solution 2: Neural Hypergraph
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_neural_hypergraph --scale_hyperedges 3

# Both solutions combined
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_cluster_pooling --use_neural_hypergraph
```

| Flag | Effect |
|------|--------|
| (none) | Original STH-SepNet baseline |
| `--use_cluster_pooling` | LLM sees C curves instead of 1 mean |
| `--use_neural_hypergraph` | Neural scoring replaces KNN hypergraph |

---

## Setup

### Environment

```bash
pip install -r requirements.txt
```

### Data

Download from [Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link), place under `./dataset/`.

### Pretrained Models

Download from HuggingFace, place under `./huggingface/`:

| Model | Parameters | dim |
|-------|-----------|-----|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased) | 110M | 768 |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | 124M | 768 |
| [LLAMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1230M | 2048 |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b) | 6740M | 4096 |
| [LLAMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8000M | 4096 |

Example directory: `huggingface/google_bert/` for BERT.

---

## Original Training (from original repo)

```bash
# STH-SepNet (full model with hypergraph, k=3)
sh ./scripts/BIKE/BERT_Bike_order3.sh

# STH-SepNet-GNN (GCN only, no hypergraph)
sh ./scripts/BIKE/BERT_Bike.sh

# STH-SepNet-Mixorder (GCN + HGCN mixed)
sh ./scripts/BIKE/BERT_Bike_mixorder3.sh
```

Also provides Windows `.bat` scripts under `./scripts/*/`.

---

## File Structure

```
├── train_sthsepnet.py                     # Single-GPU training (baseline + solutions)
├── OPEN_PROBLEM_SOLUTIONS.md              # Full solution documentation
├── models/
│   ├── ST_SepNet.py                       # Original model (unmodified)
│   ├── Solution1_ClusterPooling.py        # Solution 1 module
│   └── Solution2_NeuralHypergraphGenerator.py  # Solution 2 module
├── layers/                                # Model layers
├── data_provider/                         # Data loading
├── utils/                                 # Utilities
├── scripts/                               # Original training scripts
└── dataset/                               # (Download separately)
```

## Citation

```bibtex
@inproceedings{chen2025decoupling,
  title={Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs},
  author={Chen, Jiawen and Shao, Qi and Chen, Duxin and Yu, Wenwu},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2025},
  month={August},
  address={Toronto, Canada},
  publisher={ACM}
}
```
