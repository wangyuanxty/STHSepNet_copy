# STH-SepNet：开放问题分析与改进

基于 Chen et al., *Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs*, KDD 2025.  
[论文](https://arxiv.org/abs/2505.19620) | [原仓库](https://github.com/jiawenchen10/STHSepNet)

STH-SepNet 将时空建模解耦为两个独立模块：轻量 LLM 负责全局时间动态，自适应超图神经网络负责高阶空间交互，最后通过可学习门控融合两者的输出。原论文在五个数据集上取得了 SOTA 结果，同时在 Section 5 的 "Limitations and future work" 中指出了当前设计的两个局限。本文档围绕这两个开放问题展开分析，提出两种端到端的改进方案并给出初步实验结果。

---

## 1. 开放问题分析

### 1.1 全局均值池化导致的空间信息丢失

论文指出"框架假设时间与空间依赖可以被干净地解耦，但这在时空因果耦合的快速演化事件中不一定成立"。该问题的直接表现位于模型第 353 行：

```python
x_enc_pool = x_enc.mean(2, keepdim=True)   # (B, T, N) → (B, T, 1)
```

N 个传感器的速度在进入 LLM 之前被压缩为一条全局均值曲线。当某个路口发生事故，其速度从 60 骤降至 8，而其余路口维持在正常范围的波动——两个信号在均值中相互抵消，LLM 实际接收到的输入几乎没有反映出这一异常。随后 LLM 输出的单一预测经 `.expand` 复制 N 份分配给所有传感器，无论该位置是否受到事故影响。

论文 Figure C.1 的消融实验间接验证了仅在后端融合阶段增强交互的思路不可行：`attentiongate` 和 `lstmgate` 试图在 LLM 和 STHGNN 输出之间做跨模态交互，但两者的效果均不及最简单的 `adaptive` gate（单层线性映射加 sigmoid）。fusion 阶段两个模态已完成各自编码，此时强行引入 cross-attention 并不能弥补输入端的信息损失，反而引入冗余交互。

据此判断，问题的症结不在两个模块之间缺乏交互，而在于 LLM 接收的输入本身就丢失了空间差异性。

### 1.2 KNN 超图构建的可微性缺陷

论文同时指出"自适应超图对实时节点特征更新的依赖，在延迟敏感场景下可能带来计算开销问题"。超边构建位于 `AdaGNN.py` 第 339 行：

```python
nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
nn.fit(transformed_features.cpu().detach().numpy())
distances, indices = nn.kneighbors(...)
```

KNN 操作依赖 sklearn 实现，需要将 GPU 上的特征张量搬回 CPU，完成后再搬回 GPU。更为关键的是，超边关联矩阵 H 由硬 0/1 构成，梯度在此处断裂，超图结构的质量只能通过 Embedding 的间接更新来改善，无法接收预测损失的直接影响。

---

## 2. 方案一：Cluster Pooling

### 动机

`x_enc.mean(2)` 将 N 条传感器曲线压缩为一条，所有空间差异被抹平。现考虑用可学习的软聚类替代全局均值：将 N 个节点分配至 C 个聚类（C 远小于 N），每个聚类产出一条代表性曲线送入 LLM，LLM 输出 C 条预测后再反池化回 N 个节点。

### 方法

聚类由 STHGNN 的节点 Embedding 驱动——该 Embedding 同时也是超图构造的输入，因此聚类天然携带空间结构信息。设节点嵌入 $\mathbf{E} \in \mathbb{R}^{N \times D}$（取自 `hgc.emb1.weight`），可学习聚类中心 $\mathbf{C} \in \mathbb{R}^{C \times D}$，温度 $\tau$：

$$\mathbf{S} = \text{softmax}\left(\frac{\mathbf{E} \mathbf{C}^\top}{\tau}\right) \in \mathbb{R}^{N \times C}$$

其中 $\mathbf{S}_{i,k}$ 表示节点 $i$ 属于聚类 $k$ 的概率。池化和反池化分别为两次矩阵乘法：

$$\mathbf{X}^{\text{clustered}} = \mathbf{X}^{\text{enc}} \cdot \mathbf{S} \in \mathbb{R}^{B \times T \times C}, \quad
\mathbf{X}_{\text{per-node}} = \mathbf{X}_{\text{out}} \cdot \mathbf{S}^\top \in \mathbb{R}^{B \times P \times N}$$

为防止所有聚类中心退化至同一点（此时退化为全局均值池化），引入多样性正则项，惩罚不同聚类中心之间过高的余弦相似度：

$$\mathcal{L}_{\text{div}} = \frac{1}{C(C-1)} \sum_{i \neq j} \text{ReLU}\left(\frac{\mathbf{c}_i^\top \mathbf{c}_j}{\|\mathbf{c}_i\| \|\mathbf{c}_j\|}\right)$$

总损失：

$$
\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.1 \cdot \mathcal{L}_{\text{div}}
$$
### 实现

`soft_assign` 仅依赖权重矩阵（Embedding 与 cluster_centers），不依赖 STHGNN 的前向计算结果，LLM 与 STHGNN 保持并行。每条聚类曲线拼接一个可学习的聚类身份嵌入（`cluster_identity`），告知 LLM 其所代表的节点类型。

```python
logits = node_embeddings @ cluster_centers.T / temperature
soft_assign = F.softmax(logits, dim=-1)          # (N, C)
x_clustered = x_enc @ soft_assign                # (B, T, C)
# ... LLM 处理 ...
LLM_per_node = LLM_out @ soft_assign.T           # (B, P, N)
```

### 相关文献

STID (Shao et al., CIKM 2022) 论证了为每个时空节点引入可学习身份嵌入对预测精度的提升作用。聚类身份嵌入沿用了这一思想——与 STID 的逐节点身份不同，此处采用逐聚类身份以适配输入压缩的需求。Token Merging (Bolya et al., ICLR 2023) 指出 Transformer 中软聚类相比硬聚类更稳定，与本文采用 softmax 分配而非离散分组的做法一致。

---

## 3. 方案二：Neural Hypergraph

### 动机

KNN 超图构建的核心操作是"根据两个节点的特征向量判断它们是否应连接"。该操作可自然地替换为一个可微分的 MLP 评分网络，输入为两节点特征的拼接，输出为一个标量分数。对所有节点对评分后，通过 Gumbel-Softmax 实现可微分的 Top-K 选择，得到稀疏超边关联矩阵。

### 方法

节点特征沿用原论文编码方式：$\mathbf{h} = \tanh(\alpha \cdot \text{Linear}(\text{Emb}(\text{idx})))$。对每对节点 $(i,j)$ 计算连接分数：

$$s_{ij} = \text{MLP}\big([\mathbf{h}_i \| \mathbf{h}_j]\big)$$

得到 $\mathbf{S} \in \mathbb{R}^{N \times N}$ 后，将对角线置 $-\infty$ 以排除自连接。训练时加入 Gumbel 噪声，用 Straight-Through Estimator 实现前向硬选边（0/1 mask）与反向软梯度传播。

Gumbel-Softmax 的梯度形式为：

$$\frac{\partial p_{ij}}{\partial s_{im}} = \frac{1}{\tau} \cdot p_{ij} \cdot (\delta_{jm} - p_{im})$$

### 实现

评分网络为三层 MLP（2D → 128 → 128 → 1）。输出矩阵形状为 (N, N)，直接替代原论文的超边关联矩阵，下游的 `HypergraphConvolution` 和 `HypergraphAttention` 无需修改。GCN 分支（`gconv1/2/3`）保持不变。

```python
h_i = h.unsqueeze(1).expand(N, N, D)
h_j = h.unsqueeze(0).expand(N, N, D)
pairs = torch.cat([h_i, h_j], dim=-1)
scores = self.scorer(pairs).squeeze(-1)

if self.training:
    H = gumbel_softmax_topk(scores, max_k, temperature)
else:
    H = hard_topk(scores, max_k)
```

### 相关文献

Gumbel-Softmax (Jang et al., ICLR 2017) 为离散采样的可微重参数化提供了理论依据。GAT (Veličković et al., ICLR 2018) 证明了通过学习得到的注意力权重优于基于固定距离度量的连接——这一结论支持了用可学习评分网络替代欧氏距离 KNN 的设计。

---

## 4. 使用

```bash
# 基线（原论文）
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295

# 方案一：Cluster Pooling
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_cluster_pooling --num_clusters 8

# 方案二：Neural Hypergraph
python train_sthsepnet.py --llm_model BERT --data inflow \
  --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
  --use_neural_hypergraph --scale_hyperedges 3
```

| 参数 | 效果 |
|------|------|
| (无) | 原论文基线 |
| `--use_cluster_pooling` | 软聚类池化替代全局均值 |
| `--use_neural_hypergraph` | 神经评分网络替代 KNN |

---

## 5. 初步实验结果

BIKE-Inflow 数据集，BERT 主干，训练 5 个 epoch：

| 方法 | Test MAE | Test RMSE |
|------|---------|----------|
| 原论文 STH-SepNet (BERT) | 5.18 | 14.40 |
| 方案一 Cluster Pooling (C=8) | 5.29 | 13.86 |
| 方案二 Neural Hypergraph (K=3) | 5.13 | 13.61 |

方案二在 RMSE 上降低了 5.5%（14.40 → 13.61），MAE 也有小幅改善。方案一的 RMSE 降低了 3.8%，但 MAE 略高，可能与超参数 C=8 的选择有关。以上仅训练 5 个 epoch，完整 50 epoch 训练后的结果预期有进一步提升空间。

---

## 6. 环境配置

```bash
pip install -r requirements.txt
```

数据从 [Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link) 下载至 `./dataset/`。预训练模型从 HuggingFace 下载至 `./huggingface/`：

| 模型 | 参数量 | 维度 |
|------|:---:|:---:|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased) | 110M | 768 |
| [GPT-2](https://huggingface.co/openai-community/gpt2) | 124M | 768 |
| [LLAMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1230M | 2048 |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b) | 6740M | 4096 |
| [LLAMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8000M | 4096 |

原论文训练脚本位于 `./scripts/`，Windows 使用对应 `.bat` 文件。

---

## 7. 文件结构

```
├── train_sthsepnet.py
├── models/
│   ├── ST_SepNet.py                            # 原论文模型
│   ├── Solution1_ClusterPooling.py             # 方案一
│   └── Solution2_NeuralHypergraphGenerator.py  # 方案二
├── layers/   data_provider/   utils/   scripts/
└── dataset/                                    # 需下载
```

## 引用

```bibtex
@inproceedings{chen2025decoupling,
  title={Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs},
  author={Chen, Jiawen and Shao, Qi and Chen, Duxin and Yu, Wenwu},
  booktitle={KDD},
  year={2025}
}
```
