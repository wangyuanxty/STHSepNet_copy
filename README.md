# STH-SepNet

[KDD 2025] **Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs**

[论文](https://arxiv.org/abs/2505.19620) | [原仓库](https://github.com/jiawenchen10/STHSepNet)

**摘要**：时空预测是交通管理、气候监测等领域的关键任务。STH-SepNet 将时间建模和空间建模解耦——轻量 LLM 捕捉低秩时间动态，自适应超图神经网络捕捉高阶空间交互，门控机制融合两者。

---

## 开放问题与方案

论文第 5 节提出了两个开放问题，本仓库给出两种端到端解决方案：

### 方案一：Cluster Pooling

原论文用 `x_enc.mean(2)` 把 N 个传感器取全局均值，空间差异性被抹平。方案一用可学习软聚类替代全局均值——N 条传感器曲线压缩为 C 条代表性曲线，每条带空间身份标签，LLM 看到的不再是一条模糊均值。输出时同一软分配矩阵反池化回 N 个节点。

### 方案二：Neural Hypergraph

原论文用 sklearn KNN 构建超边，不可微且需要 CPU-GPU 数据拷贝。方案二用一个小的 MLP 评分网络替代 KNN——对每对节点输出连接分数，Gumbel-Softmax 做可微 Top-K 选边。全部 GPU 矩阵乘法，超图结构可以被预测损失直接优化。

### 使用

```bash
# 基线（原论文）
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295

# 方案一
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_cluster_pooling --num_clusters 8

# 方案二
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_neural_hypergraph --scale_hyperedges 3
```

| 参数 | 效果 |
|------|------|
| (无) | 原论文基线 |
| `--use_cluster_pooling` | LLM 看到 C 条曲线而非 1 条 |
| `--use_neural_hypergraph` | 神经评分替代 KNN |

---

## 实验结果

BIKE-Inflow, BERT 主干, 5 个 epoch：

| 方法 | Test MAE | Test RMSE |
|------|---------|----------|
| 原论文 STH-SepNet (BERT) | **5.18** | **14.40** |
| 方案一 Cluster Pooling (C=8) | 5.29 | 13.86 |
| 方案二 Neural Hypergraph (K=3) | 5.13 | 13.61 |

*仅跑了 5 epoch，跑满 50 epoch 效果应该更好。*

---

## 环境

```bash
pip install -r requirements.txt
```

## 数据

从 [Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link) 下载，放到 `./dataset/`。

## 预训练模型

从 HuggingFace 下载，放到 `./huggingface/`：

| 模型 | 参数量 | 维度 |
|------|:---:|:---:|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased) | 110M | 768 |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | 124M | 768 |
| [LLAMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1230M | 2048 |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b) | 6740M | 4096 |
| [LLAMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8000M | 4096 |

例如 BERT 放在 `huggingface/google_bert/`。

---

## 原论文训练脚本

```bash
# STH-SepNet（完整模型，超图 k=3）
sh ./scripts/BIKE/BERT_Bike_order3.sh

# STH-SepNet-GNN（仅 GCN，无超图）
sh ./scripts/BIKE/BERT_Bike.sh

# STH-SepNet-Mixorder（GCN + HGCN 混合）
sh ./scripts/BIKE/BERT_Bike_mixorder3.sh
```

Windows 用 `./scripts/*/` 下的 `.bat` 文件。

---

## 文件结构

```
├── train_sthsepnet.py                     # 单卡训练脚本（基线 + 方案）
├── OPEN_PROBLEM_SOLUTIONS.md              # 方案详细文档
├── models/
│   ├── ST_SepNet.py                       # 原模型（未修改）
│   ├── Solution1_ClusterPooling.py        # 方案一
│   └── Solution2_NeuralHypergraphGenerator.py  # 方案二
├── layers/                                # 模型层
├── data_provider/                         # 数据加载
├── utils/                                 # 工具
├── scripts/                               # 原论文训练脚本
└── dataset/                               # 数据（需下载）
```

## 引用

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
