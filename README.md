# STH-SepNet

[KDD 2025] **Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs**

[论文](https://arxiv.org/abs/2505.19620) | [原仓库](https://github.com/jiawenchen10/STHSepNet)

STH-SepNet 将时间建模和空间建模解耦——轻量 LLM 捕捉低秩时间动态，自适应超图神经网络捕捉高阶空间交互，门控机制融合两者。

---

## 论文开放问题

论文第 5 节 "Limitations and future work" 提出两个开放问题：

**问题一：解耦假设在时空因果耦合时不一定成立。**

> The framework assumes temporal and spatial dependencies can be cleanly decoupled, which may not hold in scenarios where these dimensions are intrinsically intertwined (e.g., rapidly evolving events with coupled spatio-temporal causality).

回头看代码第 353 行：

```python
x_enc_pool = x_enc.mean(2, keepdim=True)   # (B, T, 358) → (B, T, 1)
```

358 个传感器的速度被取了一个全局均值。路口 A 从 60 骤降到 8，357 个普通路口波动不大——两个信号在均值里互相抵消，LLM 看到的就是一条基本没变化的曲线。之后 LLM 输出一条全局预测，被 `.expand` 复制 358 份分给所有传感器。对事故点太乐观，对绕行点方向都是反的。

一种直觉的想法是在 LLM 和 STHGNN 的输出之间加 cross-attention 来捕捉时空交互。但论文 Figure C.1 的消融实验已经否定了这条路——`attentiongate` 和 `lstmgate` 在所有数据集上都不如最简单的 `adaptive` gate（一个线性层 + sigmoid）。fusion 阶段两个模态已经各自编码完了，硬做 cross-attention 只会引入噪声。

所以问题可能不在 LLM 和 STHGNN 之间缺交互，而在于 LLM 的输入在池化时丢失了空间差异。

**问题二：自适应超图的 KNN 构建存在可微性和效率问题。**

> The adaptive hypergraph's reliance on real-time node feature updates could pose challenges in latency-critical applications, where computational overhead for dynamic hyperedge generation might limit responsiveness.

超边构建在 `AdaGNN.py` 第 339 行：

```python
nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
nn.fit(transformed_features.cpu().detach().numpy())   # 搬到 CPU
distances, indices = nn.kneighbors(...)                 # sklearn 跑
```

两方面问题。一是不可微——超边是硬 0/1，梯度在这里断了。二是 `.cpu().detach().numpy()` 把数据从 GPU 拉到 CPU 给 sklearn，跑完再搬回 GPU。

---

## 方案一：Cluster Pooling（针对问题一）

替换 `x_enc.mean(2)`。取 C 个可学习的聚类中心（C 远小于 N），每个节点根据它的 Embedding 向量被软分配到不同聚类。同一个聚类的节点的速度曲线加权平均，形成 C 条有代表性的曲线，每条拼接一个"聚类身份嵌入"告诉 LLM 它是哪类节点。LLM 输出 C 条预测后，用同一个软分配矩阵反池化回 N 个节点。

设节点 Embedding 为 $\mathbf{E} \in \mathbb{R}^{N \times D}$（取 `hgc.emb1.weight`，和超图构造同一套），聚类中心为 $\mathbf{C} \in \mathbb{R}^{C \times D}$：

$$\mathbf{S} = \text{softmax}\left(\frac{\mathbf{E} \mathbf{C}^\top}{\tau}\right) \in \mathbb{R}^{N \times C}$$

池化：$\mathbf{X}^{\text{clustered}} = \mathbf{X}^{\text{enc}} \cdot \mathbf{S}$，从 (B,T,N) 到 (B,T,C)。反池化：$\mathbf{X}_{\text{per-node}} = \mathbf{X}_{\text{out}} \cdot \mathbf{S}^\top$，从 (B,pred_len,C) 回到 (B,pred_len,N)。

聚类多样性损失防止所有中心退化到同一点（那就退化成全局均值了）：

$$\mathcal{L}_{\text{div}} = \frac{1}{C(C-1)} \sum_{i \neq j} \text{ReLU}\left(\frac{\mathbf{c}_i^\top \mathbf{c}_j}{\|\mathbf{c}_i\| \|\mathbf{c}_j\|}\right)$$

总损失 $\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.1 \cdot \mathcal{L}_{\text{div}}$。

`soft_assign` 只依赖权重矩阵（Embedding + cluster_centers），不依赖 STHGNN 的前向结果，所以 LLM 和 STHGNN 仍然可以并行跑。

参考：
- **STID** (CIKM 2022) — 证明给每个时空节点加可学习身份嵌入对预测有明显帮助。聚类身份嵌入本质上是给每类区域挂一张身份证，让 LLM 知道它处理的是高速公路还是居民区。
- **Token Merging** (ICLR 2023) — 证明 Transformer 里软聚类比硬聚类稳定。不做离散节点分组，用 softmax 让每个节点可以部分属于多个聚类。

---

## 方案二：Neural Hypergraph（针对问题二）

替换 sklearn KNN。用一个小的 MLP 对每对节点打分（输入是两个节点特征的拼接，输出一个标量），然后用 Gumbel-Softmax 做可微的 Top-K 选边。全部是矩阵乘法，可以留在 GPU 上，超图结构可以被预测损失的梯度直接优化。

节点特征和原论文一样：$\mathbf{h} = \tanh(\alpha \cdot \text{Linear}(\text{Emb}(\text{idx})))$。对每对节点：

$$s_{ij} = \text{MLP}\big([\mathbf{h}_i \| \mathbf{h}_j]\big)$$

分数矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，对角线置 $-\infty$。训练时加 Gumbel 噪声做松弛选边，Straight-Through Estimator：前向走硬 0/1 mask，反向梯度走 soft 分布。推理时不加噪声，直接硬 Top-K。

Gumbel-Softmax 梯度：

$$\frac{\partial p_{ij}}{\partial s_{im}} = \frac{1}{\tau} \cdot p_{ij} \cdot (\delta_{jm} - p_{im})$$

评分网络是三层 MLP（2D → 128 → 128 → 1）。输出 H 形状 (N, N)，直接替代原来的超边关联矩阵，`HypergraphConvolution` / `HypergraphAttention` 不用改，GCN 分支（`gconv1/2/3`）也完全不动。

```python
h_i = h.unsqueeze(1).expand(N, N, D)
h_j = h.unsqueeze(0).expand(N, N, D)
pairs = torch.cat([h_i, h_j], dim=-1)
scores = self.scorer(pairs).squeeze(-1)         # (N, N)
scores = scores + diag(full(N, -inf))

if self.training:
    H = gumbel_softmax_topk(scores, max_k, temperature)
else:
    H = hard_topk(scores, max_k)
```

参考：
- **Gumbel-Softmax** (ICLR 2017) — 离散采样的可微重参数化。
- **GAT** (ICLR 2018) — 可学习连接权重优于固定距离度量。用评分网络替代欧氏距离 KNN，本质是把"谁该连谁"从数学公式变成可学习函数。

---

## 使用

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

*仅 5 epoch，跑满 50 epoch 效果应该更好。*

---

## 环境

```bash
pip install -r requirements.txt
```

## 数据

从 [Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link) 下载，放到 `./dataset/`。

## 预训练模型

从 HuggingFace 下载，放到 `./huggingface/`（例如 BERT 放在 `huggingface/google_bert/`）：

| 模型 | 参数量 | 维度 |
|------|:---:|:---:|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased) | 110M | 768 |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | 124M | 768 |
| [LLAMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1230M | 2048 |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b) | 6740M | 4096 |
| [LLAMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8000M | 4096 |

---

## 原论文训练脚本

```bash
sh ./scripts/BIKE/BERT_Bike_order3.sh    # STH-SepNet（完整模型，k=3）
sh ./scripts/BIKE/BERT_Bike.sh           # STH-SepNet-GNN（仅 GCN）
sh ./scripts/BIKE/BERT_Bike_mixorder3.sh # STH-SepNet-Mixorder
```

Windows 用 `./scripts/*/` 下的 `.bat`。

---

## 文件结构

```
├── train_sthsepnet.py                          # 训练脚本（基线 + 方案）
├── models/
│   ├── ST_SepNet.py                            # 原模型（未修改）
│   ├── Solution1_ClusterPooling.py             # 方案一模块
│   └── Solution2_NeuralHypergraphGenerator.py  # 方案二模块
├── layers/   data_provider/   utils/   scripts/
└── dataset/                                    # 需下载
```

## 引用

```bibtex
@inproceedings{chen2025decoupling,
  title={Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs},
  author={Chen, Jiawen and Shao, Qi and Chen, Duxin and Yu, Wenwu},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2025}
}
```
