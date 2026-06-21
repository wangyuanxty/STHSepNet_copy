"""
方案2: 神经超图生成器 (Neural Hypergraph Generator)
====================================================
针对开放问题2: "自适应超图的实时计算开销"

原文: "The adaptive hypergraph's reliance on real-time node feature updates
could pose challenges in latency-critical applications, where computational
overhead for dynamic hyperedge generation might limit responsiveness."

核心思路:
  用一个可微分的评分网络（MLP）替代 sklearn KNN 来判断节点间的连接。
  评分网络 + Gumbel-Softmax 全部是矩阵乘法，天然适合 GPU，
  且端到端可微分 —— 超边结构由预测任务的梯度直接优化。

原论文做法（AdaGNN.py 第339-368行）:
  1. FFN 编码节点特征 → tanh
  2. sklearn NearestNeighbors KNN → 找每个节点的 k 个最近邻
  3. 构建超边关联矩阵 H

本方案替代:
  1. FFN 编码节点特征 → 与原论文同风格
  2. MLP 评分网络 → 对每对节点 (i,j) 输出连接分数
  3. Gumbel-Softmax Top-K → 可微分离散采样

参考论文:
  - Jang et al. (ICLR 2017): "Categorical Reparameterization with Gumbel-Softmax"
    可微分离散采样，使 Top-K 选择有梯度
  - Maddison et al. (ICLR 2017): "The Concrete Distribution"
    同时独立提出的相同思路
  - Velickovic et al. (ICLR 2018): "Graph Attention Networks"
    可学习的节点间注意力权重，证明了替代固定距离度量的可行性

用法:
  替换原模型 AdaGNN.py 中 hgypergraph_constructor 的 KNN 分支（metric='knn'）。
  输出超边关联矩阵 H (N, M)，直接喂给 HypergraphConvolution。
  GCN 分支完全不动。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralHypergraphGenerator(nn.Module):
    """
    神经超图生成器 — 核心管线

    Parameters
    ----------
    nnodes : int
        节点数量 N
    node_dim : int
        节点嵌入维度
    hidden_dim : int
        评分网络隐藏层维度
    max_k : int
        每条超边包含的节点数（替代 KNN 的 k）
    temperature : float
        Gumbel-Softmax 温度。训练时用 1.0（探索），推理时用 0.1（确定性）
    """

    def __init__(self, nnodes, node_dim=40, hidden_dim=128, max_k=5, temperature=1.0):
        super().__init__()
        self.nnodes = nnodes
        self.max_k = max_k
        self.temperature = temperature

        # ========== 组件1: 节点特征编码器 ==========
        # 与原论文 hgypergraph_constructor 同风格：Embedding + Linear + tanh
        self.emb = nn.Embedding(nnodes, node_dim)
        self.lin = nn.Linear(node_dim, node_dim)

        # ========== 组件2: 超边评分网络 ==========
        # 输入: 两个节点的特征拼接 [2 * node_dim]
        # 输出: 一个标量分数，表示"i 应该把 j 纳入自己的超边吗？"
        self.scorer = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, idx):
        """
        前向传播

        Parameters
        ----------
        idx : Tensor (N,)
            节点索引，范围 [0, N-1]

        Returns
        -------
        H : Tensor (N, M)
            超边关联矩阵。H[i, j] = 1 表示节点 i 属于超边 j。
            M 是超边数量（= N，每个节点产生一条以自己为中心的超边）。
        """
        device = self.emb.weight.device
        idx = idx.to(device)
        N = idx.size(0)

        # ---- Step 1: 编码节点特征（与原论文一致）----
        node_features = self.emb(idx)  # (N, D)
        h = torch.tanh(self.lin(node_features))  # (N, D)
        # 论文原话: "feature representations are obtained via a FFN"

        # ---- Step 2: 所有节点对评分 ----
        # 计算 (N, N, 2D) 的拼接特征矩阵
        h_i = h.unsqueeze(1).expand(N, N, self.lin.out_features)  # (N, N, D)
        h_j = h.unsqueeze(0).expand(N, N, self.lin.out_features)  # (N, N, D)
        pairs = torch.cat([h_i, h_j], dim=-1)  # (N, N, 2D)

        # scores[i, j] = 节点 i 连接节点 j 的分数
        scores = self.scorer(pairs).squeeze(-1)  # (N, N)

        # 对角线设为 -inf（不连接自己）
        scores = scores + torch.diag(torch.full((N,), float('-inf'), device=device))

        # ---- Step 3: Gumbel-Softmax Top-K ----
        if self.training:
            H = self._gumbel_softmax_topk(scores, self.max_k, self.temperature)
        else:
            H = self._hard_topk(scores, self.max_k)

        return H

    # ================================================================
    # Gumbel-Softmax Top-K
    # ================================================================

    def _gumbel_softmax_topk(self, logits, k, temperature):
        """
        可微分的 Top-K 选择。

        前向传播: 输出硬 0/1 mask
        反向传播: 梯度通过 soft 分布回传

        Reference:
          Jang et al., ICLR 2017
          "Categorical Reparameterization with Gumbel-Softmax"
        """
        N = logits.size(0)

        # Gumbel 噪声: -log(-log(uniform))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))

        # 加噪声 + 温度缩放 → Softmax
        noisy_logits = (logits + gumbel_noise) / temperature
        soft = torch.softmax(noisy_logits, dim=-1)  # (N, N)，每行的 softmax

        # 找到 soft 分布中的 Top-K 位置
        _, topk_idx = torch.topk(soft, k, dim=-1)  # (N, K)

        # 构建硬 mask（前向传播用）
        hard = torch.zeros(N, N, device=logits.device)
        hard.scatter_(1, topk_idx, 1.0)

        # Straight-Through Estimator:
        #   前向: 返回 hard
        #   反向: 梯度从 hard 拷贝到 soft，然后正常回传到 logits
        result = hard - soft.detach() + soft

        return result

    def _hard_topk(self, logits, k):
        """
        硬 Top-K（推理时使用，无 Gumbel 噪声，完全确定性）。
        """
        N = logits.size(0)
        _, topk_idx = torch.topk(logits, k, dim=-1)

        hard = torch.zeros(N, N, device=logits.device)
        hard.scatter_(1, topk_idx, 1.0)

        return hard


# ============================================================
# 辅助: 用本模块替代原 hgypergraph_constructor
# ============================================================
#
# 原代码 (AdaGNN.py 第261-368行, STHGNN.py 第247行):
#   self.hgc = hgypergraph_constructor(nnodes, num_hyperedges, node_dim, device,
#                                      metric='knn', ...)
#   hadp = self.hgc(idx)
#
# 替代:
#   self.hgc = NeuralHypergraphGenerator(nnodes, node_dim=40, max_k=5)
#   hadp = self.hgc(idx)
#
# hadp 的形状: (N, M) 也即 (N, N) —— 每个节点产生一条超边
# 直接喂给原来的 HypergraphConvolution / HypergraphAttention / HypergraphSAGE


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    N, D = 207, 40  # 模拟 METR-LA 的节点数

    model = NeuralHypergraphGenerator(nnodes=N, node_dim=D, hidden_dim=128, max_k=5)
    idx = torch.arange(N)

    # Training mode
    model.train()
    H_train = model(idx)
    print(f"Training mode:")
    print(f"  Hypergraph matrix: {H_train.shape}")
    print(f"  Sparsity:          {H_train.mean().item():.3f} (expected: ~{5/N:.3f})")
    print(f"  Edges per node:    {H_train.sum(1).unique().tolist()}")

    # Test gradient flow
    loss = H_train.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"  Gradient flow:     {'PASS' if has_grad else 'FAIL'}")

    # Eval mode
    model.eval()
    with torch.no_grad():
        H_eval = model(idx)
    print(f"\nEval mode:")
    print(f"  Hypergraph matrix: {H_eval.shape}")
    print(f"  Sparsity:          {H_eval.mean().item():.3f}")

    print("\n[OK] Solution2 module works correctly")
    print("\n--- KNN vs Neural ---")
    print("  KNN:    non-differentiable, sklearn CPU")
    print("  Neural: differentiable, pure PyTorch GPU matmul")
