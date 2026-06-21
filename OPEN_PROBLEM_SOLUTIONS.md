# STH-SepNet 开放问题与解决方案

## 1. 两个开放问题

论文第 5 节 "Limitations and future work" 提出了两个开放问题：

第一个：

> The framework assumes temporal and spatial dependencies can be cleanly decoupled, which may not hold in scenarios where these dimensions are intrinsically intertwined (e.g., rapidly evolving events with coupled spatio-temporal causality).

翻译：框架假设时间和空间依赖可以干净地解耦，但这在时空因果耦合的快速演化事件中不一定成立。

回头看代码第 353 行，LLM 的输入是这样算的：

```python
x_enc_pool = x_enc.mean(2, keepdim=True)   # (B, T, 358) → (B, T, 1)
```

358 个传感器的速度被取了一个全局均值。路口 A 从 60 骤降到 8，357 个普通路口波动不大——这两个信号在均值里互相抵消，LLM 看到的就是一条基本没变化的曲线。之后 LLM 输出一条全局预测，被 `.expand` 复制 358 份分给所有传感器。对事故点来说太乐观，对绕行点来说方向都是反的。

一种直觉的想法是在 LLM 和 STHGNN 的输出之间加 cross-attention 来捕捉时空交互。但论文 Figure C.1 的消融实验已经否定了这条路——`attentiongate` 和 `lstmgate` 就是这个思路。结果 `attentiongate` 的全局注意力引入了冗余交互，稀疏数据下对噪声敏感；`lstmgate` 有时序能力但抓不住空间特征的动态变化。两种在所有数据集上都不如最简单的 `adaptive` gate（一个线性层+sigmoid）。这说明 fusion 阶段做交互是错的——两个模态已经各自编码完了，硬做 cross-attention 只会引入噪声。

所以问题可能不在 LLM 和 STHGNN 之间缺交互，而在于 LLM 的输入在池化时丢失了空间差异。均值操作抹平了不同位置的时序特征，修复输入端比在输出端做跨模态交互更直接。

---

第二个：

> The adaptive hypergraph's reliance on real-time node feature updates could pose challenges in latency-critical applications, where computational overhead for dynamic hyperedge generation might limit responsiveness.

翻译：自适应超图依赖实时的节点特征更新，动态超边生成的开销在延迟敏感场景下可能成为瓶颈。

超边构建在 `AdaGNN.py` 第 339 行：

```python
nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
nn.fit(transformed_features.cpu().detach().numpy())   # 搬到 CPU
distances, indices = nn.kneighbors(...)                 # sklearn 跑
```

两个问题。一是不可微——超边是硬 0/1，梯度在这里断了，超图结构的好坏收不到预测损失的反馈。二是 `.cpu().detach().numpy()` 把数据从 GPU 拉到 CPU 给 sklearn，跑完再搬回 GPU。对几百个节点的数据集来说这个开销不大，但这是架构上的瑕疵。

---

## 2. 方案概览

两个方案对应两个问题，互相独立，可以单独用也可以一起用。

方案一 (Cluster Pooling) 替换 `x_enc.mean(2)`。不是取一个全局均值，而是学到 C 个聚类中心，把 N 个节点软分配到 C 个聚类，每个聚类产出自己的代表性曲线给 LLM。聚类中心由 STHGNN 的节点 Embedding 驱动——和构建超图的是同一套 Embedding。LLM 看到的不再是一条模糊的均值，而是 C 条有空间身份的差异化曲线。输出时再通过同一个软分配矩阵反池化回 N 个节点。

方案二 (Neural Hypergraph) 替换 sklearn KNN。用一个小的 MLP 对每对节点打分（输入是两个节点特征的拼接，输出一个标量），然后用 Gumbel-Softmax 做可微的 Top-K 选边。全部是矩阵乘法，可以留在 GPU 上，超图结构可以被预测损失的梯度直接优化。

| 方案 | 针对问题 | 改什么 | 怎么做到的 |
|------|:---:|------|------|
| Cluster Pooling | 问题1 | `x_enc.mean(2)` | 软分配矩阵 N→C→N |
| Neural Hypergraph | 问题2 | `NearestNeighbors` | MLP 评分 + Gumbel-Softmax |

---

## 3. Cluster Pooling

### 思路

原论文把 358 条传感器曲线压成一条均值丢给 LLM。均值淹掉了事故点的暴跌，也淹掉了绕行点的飙升，LLM 看到的是一条基本平坦的线，什么都不知道。

方案一不取均值。取 C 个可学习的聚类中心（C 远小于 N），每个节点根据它的 Embedding 向量被软分配到不同聚类。同一个聚类的节点的速度曲线加权平均，形成 C 条有代表性的曲线。每条曲线拼接一个"聚类身份嵌入"告诉 LLM 它是哪类节点。LLM 输出 C 条预测后，用同样的软分配矩阵反池化回 N 个节点。

### 具体做法

设节点 Embedding 为 $\mathbf{E} \in \mathbb{R}^{N \times D}$（取的是 `hgc.emb1.weight`，和超图构造用的同一套），聚类中心为 $\mathbf{C} \in \mathbb{R}^{C \times D}$。软分配：

$$\mathbf{S} = \text{softmax}\left(\frac{\mathbf{E} \mathbf{C}^\top}{\tau}\right) \in \mathbb{R}^{N \times C}$$

池化就是矩阵乘法：$\mathbf{X}^{\text{clustered}} = \mathbf{X}^{\text{enc}} \cdot \mathbf{S}$，从 (B,T,N) 变成 (B,T,C)。每条聚类曲线过一个小的 patch embedding，加上聚类身份嵌入，进 LLM。LLM 输出后同样做一次反池化：$\mathbf{X}_{\text{per-node}} = \mathbf{X}_{\text{out}} \cdot \mathbf{S}^\top$，从 (B,pred_len,C) 回到 (B,pred_len,N)。

整个路径可微，$\mathbf{S}$ 是 softmax 出来的，链式法则直接通到 $\mathbf{C}$。

加了一个聚类多样性损失防止所有中心退化到一个点（那样就退化成全局均值了）：

$$\mathcal{L}_{\text{div}} = \frac{1}{C(C-1)} \sum_{i \neq j} \text{ReLU}\left(\frac{\mathbf{c}_i^\top \mathbf{c}_j}{\|\mathbf{c}_i\| \|\mathbf{c}_j\|}\right)$$

总损失 $\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.1 \cdot \mathcal{L}_{\text{div}}$。

### 实现

- `cluster_centers`: (C, D) 可学习参数，和 Embedding 在同一个空间
- `cluster_identity`: (C, D_llm) 可学习 Embedding，告诉 LLM 每条曲线代表哪类节点
- 温度 $\tau=1.0$

```python
logits = node_embeddings @ cluster_centers.T / temperature
soft_assign = F.softmax(logits, dim=-1)           # (N, C)
x_clustered = x_enc @ soft_assign                 # (B, T, N) → (B, T, C)
# ... LLM ...
LLM_per_node = LLM_out @ soft_assign.T            # (B, pred_len, N)
```

`soft_assign` 只依赖权重矩阵（Embedding + cluster_centers），不依赖 STHGNN 的前向计算结果，所以 LLM 和 STHGNN 还是可以并行跑。

### 参考

STID (CIKM 2022) 证明了给每个时空节点加可学习的身份嵌入对预测有明显帮助。我们把"空间身份"这个思想搬到了 LLM 输入端——聚类身份嵌入本质上就是给每类区域挂一张身份证，让 LLM 知道它现在处理的是高速公路还是居民区。

Token Merging (ICLR 2023) 证明了 Transformer 里软聚类比硬聚类稳定。我们不做离散的节点分组，而是用 softmax 分配让每个节点可以部分属于多个聚类。

---

## 4. Neural Hypergraph

### 思路

原论文用 sklearn KNN 搭超边。KNN 本身没什么问题，问题是它不在计算图里——梯度到 H 就断了，超图结构完全靠 Embedding 自身的更新间接优化。而且 sklearn 跑在 CPU 上，数据得搬来搬去。

方案二的意思很简单：KNN 做的事就是"看两个节点的特征向量，判断它们该不该连"。这件事可以交给一个小的神经网络做。输入是两个节点特征的拼接，输出一个分数。对所有节点对算一遍分数，然后 Gumbel-Softmax 选每行的 Top-K——选出来的就是超边。整个过程全在 GPU 上，全是矩阵乘法，梯度可以直接穿回评分网络。

### 具体做法

节点特征和原论文一样：$\mathbf{h} = \tanh(\alpha \cdot \text{Linear}(\text{Emb}(\text{idx})))$。对每对节点 (i, j)，评分网络输出：

$$s_{ij} = \text{MLP}\big([\mathbf{h}_i \| \mathbf{h}_j]\big)$$

得到分数矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，对角线置 $-\infty$。

训练时加 Gumbel 噪声做松弛选边：$\tilde{s}_{ij} = (s_{ij} + g_{ij}) / \tau$，Softmax 之后用 Straight-Through Estimator 把 soft 分布变成硬 0/1 mask——前向走硬的，反向梯度走软的。推理时不加噪声，直接硬 Top-K。

Gumbel-Softmax 的梯度：

$$\frac{\partial p_{ij}}{\partial s_{im}} = \frac{1}{\tau} \cdot p_{ij} \cdot (\delta_{jm} - p_{im})$$

### 实现

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

评分网络是一个三层 MLP（2D → 128 → 128 → 1）。输出 H 形状是 (N, N)，直接替代原来的超边关联矩阵，后续的 `HypergraphConvolution` / `HypergraphAttention` 不用改。GCN 分支（`gconv1/2/3`）也完全不动。

### 参考

Gumbel-Softmax (ICLR 2017) 提供了离散采样的可微重参数化方法。GAT (ICLR 2018) 证明了可学习的连接权重比固定距离度量好——我们用评分网络替代欧氏距离 KNN，本质上就是把"谁该连谁"这个决策从数学公式变成了可学习的函数。
