

# Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs

Jiawen Chen\*, Qi Shao\*, Duxin Chen†, Wenwu Yu†

School of Mathematics, Southeast University, Nanjing, Jiangsu, China

\* Jiawen Chen and Qi Shao contributed equally to this work.  
† Correspondence to Duxin Chen and Wenwu Yu.

## Abstract

Spatio-temporal prediction is a pivotal task with broad applications in traffic management, climate monitoring, energy scheduling, etc. However, existing methodologies often struggle to balance model expressiveness and computational efficiency, especially when scaling to large real-world datasets. To tackle these challenges, we propose **STH-SepNet** (*Spatio-Temporal Hypergraph Separation Networks*), a novel framework that decouples temporal and spatial modeling to enhance both efficiency and precision.

Therein, the temporal dimension is modeled using lightweight large language models, which effectively capture low-rank temporal dynamics. Concurrently, the spatial dimension is addressed through an adaptive hypergraph neural network, which dynamically constructs hyperedges to model intricate, higher-order interactions. A carefully designed gating mechanism is integrated to seamlessly fuse temporal and spatial representations.

By leveraging the fundamental principles of low-rank temporal dynamics and spatial interactions, STH-SepNet offers a pragmatic and scalable solution for spatio-temporal prediction in real-world applications. Extensive experiments on large-scale real-world datasets across multiple benchmarks demonstrate the effectiveness of STH-SepNet in boosting predictive performance while maintaining computational efficiency.

Code: <https://github.com/SEU-WENJIA/ST-SepNet-Lightweight-LLMs-Meet-Adaptive-Hypergraphs>

## CCS Concepts

- Computing methodologies → Spatial and physical reasoning
- Computing methodologies → Temporal reasoning

## Keywords

Spatio-Temporal Prediction, Graph Neural Networks, Large Language Models, Adaptive Hypergraph Neural Networks

## ACM Reference Format

Jiawen Chen, Qi Shao, Duxin Chen, and Wenwu Yu. 2025. *Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs*. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25). ACM, New York, NY, USA, 12 pages. <https://doi.org/10.1145/3711896.3736904>

---

# 1. Introduction

Spatio-temporal prediction serves as a fundamental component of modern data-driven decision-making, with applications in urban traffic forecasting, climate modeling, energy grid optimization, etc. Despite its broad significance, the field faces two primary challenges: accurately capturing dynamic spatial dependencies and ensuring computational scalability for large-scale real spatio-temporal datasets.

While deep learning has led to notable advancements, existing methods often struggle to achieve a balance between model expressiveness and computational efficiency.

Recent advances in graph neural networks (GNNs) and large language models (LLMs) have been extensively explored to address these challenges. GNNs are particularly effective in capturing spatial dependencies through graph-structured representations. However, their dependence on static graph topologies poses a significant constraint, impeding their capacity to accurately model dynamic, higher-order interactions.

For instance, in traffic networks, the influence between regions is constantly evolving and influenced by external conditions—factors that are inadequately captured by static adjacency matrices. Meanwhile, LLMs, which distinguish themselves in temporal prediction due to their strong sequence modeling capabilities, incur substantial computational expenses when applied to large node sets. Furthermore, their ability to leverage spatial structures remains limited.

The dominant approach of jointly modeling spatial and temporal features within a single framework has been shown to exacerbate these challenges, often leading to overparameterized models that are computationally demanding and difficult to optimize, without yielding proportional performance improvements. This raises a critical question:

> Can spatial and temporal modeling be decoupled to achieve both efficiency and accuracy?

![Figure 1: Spatio-temporal data and adaptive hypergraph](figure-1-placeholder)

**Figure 1:**  
(a) Spatio-temporal data exhibit spatial distribution shifts across different nodes.  
(b) Dynamic adaptive hypergraph captures evolving spatial distribution patterns.

To tackle these challenges, this work proposes a separation strategy based on two key insights.

First, temporal dynamics in spatiotemporal systems often exhibit a low-rank structure, implying that the evolution of system states can be efficiently characterized by a small number of latent factors. This low-rank property facilitates the use of lightweight sequence models, such as distilled versions of LLMs, to capture temporal trends without compromising expressiveness.

Second, spatial dependencies in spatiotemporal systems can be viewed as a form of spatial drift, where the influence between nodes shifts over time due to external factors or intrinsic system dynamics. Traditional GNNs, despite their effectiveness in many applications, struggle to capture dynamic drift due to their dependence on static graph structures.

To address this limitation, we propose an adaptive hypergraph framework that possesses enhanced representational capabilities for graphs, enabling it to model evolving higher-order interactions. This framework allows hyperedges to dynamically encapsulate shifting relationships among multiple nodes, thereby accurately reflecting the evolving nature of these connections.

Building on these insights, we propose **STH-SepNet** (*Spatio-Temporal Hypergraph Separation Network*), a novel framework that specifically integrates lightweight temporal modeling with adaptive spatial modeling.

For temporal modeling, compact LLMs such as BERT and GPT-2 are employed to efficiently capture low-rank temporal dynamics. For spatial modeling, an adaptive hypergraph neural network is introduced to dynamically construct hyperedges, enabling the representation of spatial drift and higher-order interactions. A gating mechanism is further designed to fuse temporal and spatial representations, ensuring seamless integration while maintaining computational efficiency.

The main contributions of this work are summarized as follows:

- We propose a lightweight spatio-temporal separation framework, **STH-SepNet**, for spatio-temporal prediction tasks. The framework integrates textual information and latent spatial dependencies, resulting in significant improvements in predictive performance.
- We design an adaptive hypergraph structure for spatial modeling, which dynamically constructs complex dependency relationships and enhances the extraction of spatial features through effective order modeling.
- We conduct extensive experiments to validate STH-SepNet, demonstrating state-of-the-art performance across multiple benchmarks. The proposed method demonstrates efficient execution on a single A6000 GPU, underscoring its practical applicability for real-world deployment.

---

# 2. Related Works

## 2.1 Large Models for Prediction

Large language models, recognized for their extensive parameter sizes and strong generalization capabilities, have been increasingly applied to time-series analysis tasks, including prediction, classification, and imputation.

To bridge the gap between numerical data and the text-based processing paradigm of LLMs, researchers have explored novel data formatting techniques. For instance, PromptCast converts numerical sequences into natural language prompts, while Gruver et al. encode time-series data as digit strings to enable zero-shot predictions.

These approaches demonstrate the potential of LLMs in temporal modeling while also highlighting the need for specialized adaptations to address the unique challenges of time-series data, such as irregular time intervals and long-range dependencies.

Recent efforts have sought to refine tokenization and embedding strategies to improve LLMs’ suitability for forecasting tasks. LLM4TS employs parameter-efficient fine-tuning (PEFT) to adapt pre-trained LLMs for time-series prediction, while Zhou et al. propose a unified framework for handling diverse time-series tasks. Additionally, advancements such as reprogramming frameworks and contrastive embedding strategies further align numerical and textual modalities, enhancing LLMs’ capability to process temporal data.

However, these methods predominantly focus on temporal modeling while largely overlooking spatial dependencies, which poses a fundamental limitation for spatio-temporal prediction tasks.

The integration of LLMs with transformer-based architectures has further expanded their applicability to domain-specific challenges. Models such as UniST and OpenCity incorporate transformers with graph neural networks to capture spatio-temporal dependencies, while ClimaX and Pangu-Weather illustrate the versatility of transformer-based designs in climate forecasting.

Despite these advancements, existing approaches often struggle to effectively balance spatial and temporal modeling, resulting in increased computational complexity without proportional performance improvements. This underscores the need for novel frameworks that integrate spatio-temporal structures more efficiently while maintaining computational feasibility.

## 2.2 Spatio-Temporal Prediction

Recent advances in spatio-temporal prediction have been driven by the integration of transformer-based models and graph neural networks, aimed at addressing the dual challenges of capturing long-range temporal dependencies and complex spatial interactions.

Transformer-based models, such as DLinear and TimesNet, have demonstrated strong performance in time-series forecasting by leveraging multi-scale temporal patterns and efficient attention mechanisms. Similarly, PatchTST introduces patch-based attention to enhance local dependency modeling, while iTransformer reconfigures the transformer architecture for multivariate time-series modeling.

However, these models often struggle with distribution shifts, such as changes in traffic patterns or external conditions, limiting their robustness in real-world applications.

In the realm of spatial modeling, GNNs and hypergraph structures have emerged as effective tools for capturing complex spatial relationships. STG-NCDE integrates neural controlled differential equations with GNNs to model continuous-time dynamics, while STAEformer incorporates spatial and temporal attention mechanisms within a transformer-based framework. Hypergraph-based models, such as STHGCN and GPT-ST, further advance the field by dynamically capturing higher-order dependencies and spatial drift.

However, these approaches frequently rely on static or predefined structures, limiting their ability to adapt to evolving spatial relationships and distribution shifts over time.

A key limitation of existing methods is their inability to effectively model distribution shifts, which are inherent in spatio-temporal systems. For example, STID simplifies spatio-temporal prediction but lacks adaptability to dynamic spatial interactions, while FEDformer and Autoformer primarily focus on temporal modeling without accounting for spatial distribution shifts.

To address these challenges, we propose STH-SepNet, a lightweight framework that leverages adaptive hypergraphs to model distribution shifts and lightweight transformers to capture temporal dynamics, achieving state-of-the-art performance.

---

# 3. Preliminaries

## 3.1 Problem Formulation

Given a graph set denoting the spatial feature:

$$
G = (V, E)
$$

where $V$ and $E$ represent the set of $N = |V|$ vertices and the set of edges, respectively.

The spatio-temporal prediction problem of multivariate time-series forecasting is defined as follows: suppose the historical observations from $L$ previous moments are:

$$
X^{(t-L+1):t} \in \mathbb{R}^{L \times N \times F}
$$

our model STH-SepNet aims to predict the values for the next $H$ timestamps:

$$
\hat{X}^{(t+1):(t+H)} \in \mathbb{R}^{H \times N \times F}
$$

That is:

$$
\hat{X}^{(t+1):(t+H)}
=
\mathrm{STH\text{-}SepNet}_{\theta}
\left(
X^{(t-L+1):t}, \hat{A}, \Phi
\right)
\tag{1}
$$

where $\hat{A}$ is the structural adjacency matrix, $\Phi$ is the prompt information for the input vector, and $\theta$ denotes the model parameters.

## 3.2 Adaptive Network Construction

We introduce an adaptive adjacency matrix $\tilde{A}_{adp}$ as input to the ST-Block, aiming to mitigate similarity between adjacent nodes.

Given node features:

$$
E_1, E_2 \in \mathbb{R}^{N \times d}
$$

we employ a shared-parameter feed-forward neural network (FFN) to generate node embeddings, which are then mapped to:

$$
F_1, F_2 \in \mathbb{R}^{N \times N}
$$

as follows:

$$
F_1 = \tanh(\alpha \mathrm{FFN}(E_1))
\tag{2}
$$

$$
F_2 = \tanh(\alpha \mathrm{FFN}(E_2))
\tag{3}
$$

where $\alpha$ is a scaling factor that modulates the saturation rate of the activation function.

The discrepancy between $F_1$ and $F_2$ captures directional relationships between nodes. To introduce non-linearity, we construct an asymmetric adjacency matrix:

$$
A_{adp} \in \mathbb{R}^{N \times N}
$$

as:

$$
A_{adp}
=
\mathrm{ReLU}
\left(
\tanh
\left(
\alpha
\left(
F_1^\top F_2 - F_2^\top F_1
\right)
\right)
\right)
\tag{4}
$$

This formulation effectively models asymmetric dependencies by leveraging learned node embeddings in the graph structure.

## 3.3 Incident Matrix

We integrate static spatial topology, e.g., geographic location information in a traffic network, as the static input to build an adjacency matrix $A$, defining node similarity via a negative exponential function of pairwise Euclidean distances.

The similarity $A_{ij}$ between nodes $i$ and $j$ is defined as:

$$
A_{ij}
=
\exp
\left(
-
\frac{d_{ij}^{2}}{\sigma^{2}}
\right)
$$

where $d_{ij}$ is the distance between nodes $i$ and $j$, and $\sigma$ is a scaling parameter that regulates the effect of distance on similarity. A fixed threshold is applied to maintain the sparsity of the adjacency matrix.

## 3.4 Adaptive HyperGraph Construction

**Definition 1. Hypergraph**

A high-order graph $H(V,E)$ is defined by a set of $n$ hypernodes:

$$
V = \{v_1, v_2, \cdots, v_n\}
$$

and a set of $m$ hyperedges:

$$
E = \{e_1, e_2, \cdots, e_m\}
$$

where:

$$
e_j = \left(v_1^{(j)}, \cdots, v_k^{(j)}\right)
$$

is an unordered set of nodes on hyperedge $e_j$, with $k = |e_j|$ denoting the number of nodes in the hyperedge.

**Theorem 1.**

For any $k \ge 2$, the $(k-1)$-hops neighborhood of a node $v$, denoted as $N_{k-1}(v)$, corresponds to all nodes involved in the $k$-order hyperedges in $H_v^k$, if and only if the following conditions are satisfied:

1. **Local Connectivity Condition:**  
   For each $w \in N_{k-1}(v)$, there exists at least one path from $v$ to $w$ consisting of at most $k-1$ hyperedges.

2. **Hyperedge Coverage Condition:**  
   There exists a $k$-order hyperedge $e \in H_v^k$ such that $w \in e$, and $e$ contains $v$, $w$, and at most $k-2$ intermediate nodes.

3. **Uniqueness Condition:**  
   If there exist multiple $k$-order hyperedges containing both $v$ and $w$, then these hyperedges must share the same set of intermediate nodes.

Formally:

$$
w \in N_{k-1}(v)
\Longleftrightarrow
\{v, F_1, F_2, \ldots, u_k, w\} \in H_v^k
\tag{5}
$$

where $F_1, F_2, \ldots, u_k$ are intermediary nodes.

In constructing a $(k+1)$-order hypergraph from $k$-hop neighborhoods, each node is interconnected with all nodes within its $k$-hop distance, forming one or more hyperedges.

Given a node $v_i$, its $k$-hop neighborhood $N_k(v_i)$ includes all nodes reachable within $k$ edges. Similarly, for node features $E_3$, feature representations $F_3$ are obtained via a feedforward neural network:

$$
F_3 = \tanh(\alpha \mathrm{FFN}(E_3))
\tag{6}
$$

Higher-order relationships are then constructed using K-Nearest Neighbors (KNN) on feature representations:

$$
F_3 = [f_1, f_2, \ldots, f_n]
$$

For each node $v_i$, its nearest $k$ neighbors $N(v_i)$ form a hyperedge:

$$
e_i = \{v_i\} \cup N(v_i)
$$

where:

$$
k = \max_j |e_j|
$$

is the hyperedge order and remains a predefined constant for consistency.

To determine the hypergraph adaptive adjacency matrix:

$$
H_{adp} \in \mathbb{R}^{n \times m}
$$

where $n$ is the number of nodes and $m$ is the number of hyperedges, the adjacency matrix is defined as:

$$
H_{adp,ij}
=
\begin{cases}
1, & \text{if } v_i \in e_j \\
0, & \text{otherwise}
\end{cases}
$$

Traditional graph-based methods primarily focus on pairwise interactions between nodes, which can be insufficient for modeling multiple nodes interacting simultaneously. By contrast, hypergraphs allow for the representation of higher-order interactions through hyperedges. The adaptive adjacency matrix for hypergraphs can degenerate into traditional graphs, but it leverages this flexibility to capture richer and more complex relationships.

---

# 4. Methodology

In this work, we propose **STH-SepNet**, a spatio-temporal forecasting model that integrates a pre-trained LLM with adaptive hypergraphs.

As shown in Figure 2, STH-SepNet comprises two key components:

1. Lightweight large language models for temporal dynamics.
2. Adaptive hypergraphs for spatial dependencies.

![Figure 2: Framework of STH-SepNet](figure-2-placeholder)

**Figure 2:** The framework of STH-SepNet. Given a traffic network $G=(V,E)$ and time series $X$ as an example of spatio-temporal datasets:

1. Tokenize and embed $X$ using a customized embedding layer, reprogramming with condensed text prototypes for modality alignment.
2. Incorporate dataset descriptions, task instructions, and statistical characteristics as prompt prefixes to guide input transformation.
3. Leverage a Hypergraph Spatio-Temporal module to model complex spatial dependencies and node-level variations via hierarchical representation learning.
4. Use incident matrix from real geographic network; if not available, adaptive graph or adaptive hypergraph is used.

By integrating these components, STH-SepNet generates forecasts.

## 4.1 Global Trend Module

### Local Aggregation Module

This module processes the model input node features:

$$
X \in \mathbb{R}^{B \times N \times T \times F}
$$

by executing an average pooling operation to extract the common features of all nodes within a region, capturing the overall fluctuation trends:

$$
X_{pool} = \mathrm{AvgPool}(X)
\tag{7}
$$

where:

$$
X_{pool} \in \mathbb{R}^{B \times T \times F}
$$

$B$ is the batch size, $T$ denotes the time steps, and $F=1$.

The time series embedding module reduces computational time and memory complexity by aggregating information across adjacent time steps and utilizing temporal patches. Specifically, $X_{pool}$ is partitioned into overlapping or non-overlapping blocks:

$$
X_P \in \mathbb{R}^{P \times N_P}
$$

where $P$ is the window length and the number of sliding windows is computed as:

$$
N_P = \frac{T-P}{S} + 2
$$

with stride $S$.

Each patch is treated as a time series token and embedded to obtain:

$$
\hat{X}_P \in \mathbb{R}^{P \times d_m}
$$

where $d_m$ is the hidden dimension of the LLM.

## 4.2 Prompt Adaptation Module

Given that LLMs are primarily trained on extensive text corpora and lack inherent time series knowledge, we propose a cross-modal alignment strategy that transforms time-series data into textual tokens, enabling LLMs to leverage their reasoning capabilities for specialized forecasting tasks.

To enhance predictive accuracy, we adopt Pattern-Exploiting Training, which formulates natural language templates as prompts within the embedding space. These prompts integrate three key components:

- Dataset description
- Task instructions
- Statistical characteristics

We further prepend this structured information as a prefix prompt, concatenating it with aligned temporal embeddings before feeding it into the LLM. This enables the model to generate valid outputs and adapt to downstream tasks.

### LLM Module

We utilize a partially frozen pre-trained LLM to capture temporal dependencies in traffic data, fine-tuning its feed-forward layers via LoRA.

Pretrained models serve as the backbone for STH-SepNet, comprising stacked transformer decoder modules with $N$ layers.

The input to each layer is represented as:

$$
z = \{z_1, z_2, \ldots, z_N\}
$$

where $z_1$ consists of concatenated prompt and time-series embeddings.

For the $i$-th layer, the input $z_i$ undergoes multi-head self-attention (MHSA) followed by layer normalization (LN), producing an intermediate state $\tilde{z}_i$, which is further processed by a feed-forward network and another layer normalization step to yield $z_{i+1}$.

It can be summarized as:

$$
(Q_i, K_i, V_i)
=
(W_i^Q z_i, W_i^K z_i, W_i^V z_i)
\tag{8}
$$

$$
\mathrm{head}_i
=
\mathrm{softmax}
\left(
\frac{Q_i K_i^\top}{\sqrt{d}}
\right)
V_i
\tag{9}
$$

$$
\mathrm{MHSA}(z_i,z_i,z_i)
=
W(\mathrm{head}_1 \| \cdots \| \mathrm{head}_h)
\tag{10}
$$

$$
\tilde{z}_i
=
\mathrm{LN}
\left(
z_i + \mathrm{MHSA}(z_i,z_i,z_i)
\right)
\tag{11}
$$

$$
z_{i+1}
=
\mathrm{LN}
\left(
\tilde{z}_i + \mathrm{FFN}(\tilde{z}_i)
\right)
\tag{12}
$$

where $z_i$ is the input hidden state at the $i$-th layer, and $\tilde{z}_i$ is the intermediate state after MHSA and layer normalization.

Since the LLM module outputs token sequences, we apply a linear layer to align learned patch representations. Additionally, pretrained models such as BERT, GPT-2, LLAMA, and DeepSeek can be employed for autoregressive token prediction.

## 4.3 Hypergraph Spatio-Temporal Module

To enhance spatio-temporal prediction by incorporating higher-order coupling relationships, STH-SepNet consists of four key components:

- Mixed Multi-Layer Information Aggregation Module
- Adaptive Graph Convolution Network
- Hypergraph Convolutional Network
- Temporal Convolutional Network

### Mixed Multi-Layer Information Aggregation Module

Given an input:

$$
X^{(0)} \in \mathbb{R}^{N \times C}
$$

where $N$ represents the number of nodes and $C$ is the feature dimension, the module updates node features through $k$-layer propagation.

The feature propagation at layer $k+1$ is formulated as:

$$
X^{(k+1)}
=
\alpha X^{(k)}
+
(1-\alpha)\hat{A}X^{(k)}
\tag{13}
$$

where $X^{(k)}$ is the node feature matrix at the $k$-th layer, and:

$$
\hat{A}
=
D^{-\frac{1}{2}}(A+I)D^{-\frac{1}{2}}
$$

is the normalized adjacency matrix. $D$ is the degree matrix, $I$ is the identity matrix, and $\alpha$ controls the weight of residual connections.

To further regulate the flow of information, the MixProp module incorporates a gating mechanism:

$$
G^{(k)} = \sigma(W_g X^{(k)})
$$

$$
X^{(k+1)}
=
G^{(k)} \odot X^{(k)}
+
(1-G^{(k)}) \odot \hat{A}X^{(k)}
\tag{14}
$$

where $W_g \in \mathbb{R}^{C \times C}$, $\sigma(\cdot)$ is the sigmoid activation function, and $\odot$ denotes element-wise multiplication.

The MixProp module employs $k$-layer propagation to expand the receptive field and capture dependencies between distant nodes.

### Adaptive Graph Convolution Network Module

To capture both structural and adaptive spatial dependencies, we apply three MixProp-based graph convolutions on the adaptive adjacency matrix $A_{adp}$ and the real road network $A$.

These operations extract first-order, transposed first-order, and real relationships:

$$
X_{gconv1}
=
\mathrm{MixProp}(X,A_{adp},K,\alpha)
\tag{15}
$$

$$
X_{gconv2}
=
\mathrm{MixProp}(X,A_{adp}^{T},K,\alpha)
\tag{16}
$$

$$
X_{gconv3}
=
\mathrm{MixProp}(X,A,K,\alpha)
\tag{17}
$$

where:

$$
X \in \mathbb{R}^{B \times N \times T}
$$

represents the input time-series features, $K$ is the number of propagation layers, and $\alpha$ controls residual weighting.

The final spatial representation is obtained by fusing these outputs:

$$
X_{GCN}
=
X_{gconv1}
+
X_{gconv2}
+
X_{gconv3}
\tag{18}
$$

### Adaptive Hypergraph Convolution Network Module

Given an input feature matrix:

$$
X \in \mathbb{R}^{B \times N \times T}
$$

and an adaptive hypergraph adjacency matrix $H_{adp}$, this module employs two information aggregation mechanisms:

1. Node-to-hyperedge aggregation.
2. Hyperedge-to-node aggregation.

Initially, a feedforward neural network transforms the feature matrix:

$$
X_{enc} = \mathrm{FFN}(X)
\tag{19}
$$

In the node-to-hyperedge process, each hyperedge $e_j$ accumulates information from its associated nodes $N(e_j)$, followed by hyperedge aggregation and linear transformation:

$$
X_{enc}^{e}
=
\sigma
\left(
\sum_{i \in N(e)}
H_{adp,i}
X_{enc}^{i}
W
\right)
\tag{20}
$$

where $W$ represents the trainable parameter matrix, and $\sigma(\cdot)$ is the ReLU activation function.

Subsequently, features of all hyperedges containing a node $v_i$ are aggregated back to node representation:

$$
X_{enc}^{v}
=
\sum_{j \in \mathcal{E}(v_i)}
H_{adp,j}
X_{enc}^{e_j}
\tag{21}
$$

where $\mathcal{E}(v_i)$ indicates the set of hyperedges for node $v_i$.

The output of the hypergraph spatial learning module is:

$$
X_{HGCN} = X_{enc}^{v}
$$

To integrate pairwise and high-order spatial features, we fuse representations from GCNs and HGCNs:

$$
X
=
\gamma X_{GCN}
+
(1-\gamma)X_{HGCN},
\quad
\gamma \in [0,1]
\tag{22}
$$

where $\gamma$ is a tunable parameter. When $\gamma = 1$, the model degenerates into a standard spatial learning module, whereas $\gamma = 0$ captures only higher-order dependencies.

### Spatial-Temporal Convolution Module

The Spatio-Temporal Convolution Module consists of multiple stacked ST-Blocks, each containing an S-Block for spatial dependencies and a T-Block for temporal dependencies.

In the S-Block, each node state $h_t^{(v)}$ is initialized as the input features:

$$
X \in \mathbb{R}^{B \times N \times T \times F}
$$

and updated by aggregating features from its neighbors:

$$
m_t^v
=
\sum_{u \in N(v)}
h_{t-1}^{u}
\tag{23}
$$

$$
h_t^v
=
\sigma
\left(
(1+\epsilon)h_{t-1}^{v}
+
m_t^v
\right)
\tag{24}
$$

where $m_t^v$ is the aggregated neighborhood feature at time $t$, $\sigma$ is an activation function, and $\epsilon$ is a learnable parameter.

The T-Block comprises 1-D dilated convolution layers with a gating mechanism featuring only an output gate. Given the input:

$$
\chi \in \mathbb{R}^{T \times N \times F}
$$

the gated output $h$ is defined as:

$$
h
=
\tanh(q(\chi))
\odot
\sigma(q(\chi))
\tag{25}
$$

where $q(\chi)$ is the output of the dilated convolution layers, $\odot$ denotes the Hadamard product, and $\sigma$ is the sigmoid activation function.

## 4.4 Gated Fusion Module

To integrate global trends and node heterogeneity, we fuse outputs from the pre-trained LLM and adaptive high-order spatial module, denoted as:

$$
O_1, O_2 \in \mathbb{R}^{B \times T \times N}
$$

A feedforward neural network maps the concatenated output vectors to gates of equivalent dimensions.

The gating process is formulated as:

$$
\mathrm{Gate}
=
\sigma
\left(
\mathrm{FFN}([O_1,O_2])
\right)
\tag{26}
$$

where $\sigma$ represents the sigmoid activation function, and $[\cdot,\cdot]$ denotes concatenation.

The ultimate gated fusion process can be expressed as:

$$
\tilde{O}
=
O_1 \odot \mathrm{Gate}
+
O_2 \odot (1-\mathrm{Gate})
\tag{27}
$$

where $\mathrm{Gate}$ signifies the gate map, and:

$$
\tilde{O} \in \mathbb{R}^{B \times T \times N}
$$

represents the resultant fused output.

---

# 4.5 Experiment Settings

## Datasets

We conduct experiments on five datasets:

- BIKE-Inflow
- BIKE-Outflow
- PEMS03
- BJ500
- METR-LA

The datasets are partitioned into train, validation, and test sets by the ratio of 7:1:2.

## Baselines

To evaluate effectiveness, STH-SepNet is compared with several state-of-the-art time series prediction models:

- Autoformer
- Informer
- FEDformer
- DLinear
- TimesNet
- PatchTST
- iTransformer
- TimeLLM
- AdaMSHyper

Additionally, comparisons include spatio-temporal prediction models:

- AGCRN
- MSTGCN
- MTGNN
- STGODE
- STSGCN
- STGCN
- GMAN
- STAEformer
- STD-MAE

---

# 4.6 Main Results

## 4.6.1 Effectiveness of STH-SepNet

Table 1 presents the performance of the proposed STH-SepNet method integrating pretrained model BERT across five datasets. Our method achieves the best Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) results, which can be attributed to its spatio-temporal separation strategy and adaptive hypergraph structure.

### Decoupled Spatio-Temporal Modeling

STH-SepNet addresses the limitations of joint spatio-temporal modeling by isolating temporal and spatial dependencies.

On the BIKE-Outflow dataset, where non-stationary spatial events intersect with periodic temporal trends, our method achieves:

- MAE: 5.33
- RMSE: 14.23

It outperforms joint modeling frameworks such as TimesNet and PatchTST.

### Adaptive Hypergraphs for Dynamic Spatial Drift

The proposed adaptive hypergraph structure models evolving spatial dependencies.

On the PEMS03 dataset, our method reduces RMSE to 34.17, a 28.8% improvement over dynamic graph-based approaches like STAEformer.

The dynamic hyperedge generation mechanism allows the method to adjust node relationships dynamically, capturing spatial drift.

### Scalable Efficiency

The approach significantly reduces computational complexity by decoupling the node dimension, transforming spatio-temporal prediction into parallelizable univariate tasks.

On METR-LA, STH-SepNet achieves:

- MAE: 9.42
- RMSE: 16.41

Compared to LLM-based baselines like TimeLLM, the method reduces MAE by 23.8%.

## Table 1. Performance Comparison

Prediction horizon: 48 time steps. Lookback window: $T=48$. LLM backbone: BERT.

| Model | BIKE-Inflow MAE | BIKE-Inflow RMSE | BIKE-Outflow MAE | BIKE-Outflow RMSE | PEMS03 MAE | PEMS03 RMSE | BJ500 MAE | BJ500 RMSE | METR-LA MAE | METR-LA RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoformer | 7.01 | 17.52 | 7.19 | 17.75 | 44.87 | 70.84 | 10.79 | 16.06 | 12.47 | 20.04 |
| Informer | 8.25 | 20.37 | 9.21 | 21.50 | 33.72 | 52.15 | 7.58 | 11.96 | 14.50 | 20.35 |
| FEDformer | 6.28 | 16.30 | 6.56 | 16.67 | 35.00 | 50.84 | 10.77 | 15.99 | 12.35 | 18.79 |
| DLinear | 5.71 | 15.49 | 5.82 | 15.36 | 45.30 | 66.81 | 8.55 | 13.49 | 10.90 | 17.31 |
| TimesNet | 5.54 | 15.41 | 5.56 | 15.18 | 37.54 | 62.99 | 8.67 | 13.96 | 10.22 | 18.29 |
| PatchTST | 5.53 | 15.39 | 5.63 | 15.23 | 48.42 | 78.24 | 8.79 | 14.28 | 10.13 | 18.27 |
| iTransformer | 6.05 | 16.39 | 6.15 | 16.69 | 43.63 | 70.61 | 9.01 | 14.32 | 10.15 | 18.36 |
| TimeLLM | 6.81 | 16.72 | 6.93 | 16.30 | 32.62 | 49.77 | 7.25 | 11.58 | 12.36 | 18.53 |
| AdaMSHyper | 6.72 | 16.91 | 7.04 | 17.14 | 33.49 | 50.37 | 7.41 | 11.60 | 12.51 | 18.60 |
| AGCRN | 6.64 | 16.14 | 6.77 | 16.36 | 33.14 | 54.88 | 6.32 | 12.81 | 11.39 | 23.15 |
| ASTGCN | 6.66 | 15.87 | 6.26 | 14.48 | 30.65 | 53.96 | 6.34 | 11.34 | 10.54 | 22.76 |
| MSTGCN | 5.91 | 14.11 | 6.04 | 14.24 | 29.57 | 47.97 | 5.62 | 11.15 | 10.17 | 20.24 |
| MTGNN | 6.16 | 14.80 | 5.93 | 13.93 | 29.04 | 50.32 | 5.86 | 10.91 | 9.98 | 21.23 |
| STGODE | 6.77 | 15.93 | 6.82 | 15.50 | 33.39 | 54.16 | 6.44 | 12.14 | 11.48 | 22.85 |
| STSGCN | 6.73 | 15.89 | 6.58 | 15.36 | 34.23 | 58.07 | 6.40 | 12.03 | 11.07 | 22.79 |
| STGCN | 7.08 | 15.72 | 7.36 | 16.11 | 36.02 | 53.44 | 6.73 | 12.62 | 12.38 | 22.55 |
| GMAN | 6.73 | 15.60 | 6.94 | 15.84 | 33.96 | 53.02 | 6.41 | 12.40 | 11.69 | 22.37 |
| STAEformer | 5.97 | 14.57 | 6.17 | 14.70 | 29.62 | 48.03 | 5.79 | 10.42 | 9.91 | 21.17 |
| STD-MAE | 6.13 | 14.87 | 6.21 | 14.37 | 30.40 | 48.38 | 5.92 | 11.49 | 10.52 | 23.11 |
| **STH-SepNet** | **5.18** | 14.40 | **5.33** | 14.23 | **21.03** | **34.17** | **5.58** | **9.77** | **9.42** | **16.41** |

## 4.6.2 Effectiveness of Adaptive Hypergraph Structure

Table 2 compares different graph representation strategies:

- Static graph
- Adaptive graph convolutional network
- Adaptive hypergraph structure

The results indicate that STH-SepNet achieves the best performance.

### Dynamic Adaptation to Complex Spatial Dependencies

The adaptive hypergraph dynamically adjusts spatial dependencies. In transportation networks, static graphs or standard GNNs rely on predefined adjacency relationships and fail to capture sudden disruptions such as traffic accidents.

On BIKE-Outflow, STH-SepNet reduces prediction errors:

- STH-SepNet: MAE 5.33
- Static variant: MAE 6.34

### Enhancing Lightweight Models to Surpass Large Counterparts

Large models like TimeLLM rely on massive parameters, but lightweight LLMs integrated with adaptive hypergraphs outperform them in spatio-temporal prediction.

For example, on PEMS03:

- STH-SepNet with BERT: RMSE 34.17
- TimeLLM: RMSE 50.39

### Cross-Architecture Consistency and Higher-Order Interaction Modeling

The adaptive hypergraph design is architecture-agnostic and delivers consistent performance across diverse LLM backbones, including BERT, GPT, LLAMA, and DeepSeek.

Traditional GNNs capture only pairwise node interactions, whereas hypergraphs connect multiple nodes via hyperedges, modeling multi-region joint influences.

## Table 2. Comparison with Different LLMs and Spatial Structures

| Backbone | Model | BIKE-Outflow MAE | BIKE-Outflow RMSE | PEMS03 MAE | PEMS03 RMSE |
|---|---|---:|---:|---:|---:|
| BERT | TimeLLM | 6.74 | 16.13 | 32.68 | 50.39 |
| BERT | STH-SepNet-Static | 6.34 | 16.41 | 29.53 | 48.94 |
| BERT | STH-SepNet-GNN | 5.47 | 14.36 | 21.11 | 34.52 |
| BERT | **STH-SepNet** | **5.33** | **14.23** | **21.03** | **34.17** |
| GPT2 | TimeLLM | 6.93 | 16.30 | 32.63 | 49.83 |
| GPT2 | STH-SepNet-Static | 6.53 | 16.61 | 30.03 | 49.07 |
| GPT2 | STH-SepNet-GNN | 5.68 | 14.48 | 21.85 | 35.78 |
| GPT2 | **STH-SepNet** | **5.31** | **14.24** | **21.43** | **35.01** |
| GPT3 | TimeLLM | 7.01 | 16.63 | 34.85 | 51.89 |
| GPT3 | STH-SepNet-Static | 6.79 | 16.71 | 30.64 | 49.16 |
| GPT3 | STH-SepNet-GNN | 5.66 | 13.90 | 21.27 | 34.78 |
| GPT3 | **STH-SepNet** | **5.24** | **14.16** | **21.13** | **34.69** |
| LLAMA1B | TimeLLM | 7.10 | 16.74 | 34.06 | 51.59 |
| LLAMA1B | STH-SepNet-Static | 6.27 | 16.40 | 29.76 | 48.63 |
| LLAMA1B | STH-SepNet-GNN | 5.87 | 14.80 | 22.19 | 35.87 |
| LLAMA1B | **STH-SepNet** | **5.29** | **14.20** | **21.37** | **34.92** |
| LLAMA7B | TimeLLM | 6.95 | 16.41 | 34.17 | 52.47 |
| LLAMA7B | STH-SepNet-Static | 6.73 | 16.87 | 30.24 | 49.92 |
| LLAMA7B | STH-SepNet-GNN | 5.91 | 14.94 | 21.64 | 35.24 |
| LLAMA7B | **STH-SepNet** | **5.34** | **14.31** | **21.52** | **35.17** |
| LLAMA8B | TimeLLM | 7.02 | 16.55 | 35.72 | 51.97 |
| LLAMA8B | STH-SepNet-Static | 6.85 | 16.64 | 30.47 | 49.98 |
| LLAMA8B | STH-SepNet-GNN | 5.70 | 14.27 | 21.57 | 35.23 |
| LLAMA8B | **STH-SepNet** | **5.28** | **14.20** | **21.51** | **35.19** |
| DeepSeek1.5B | TimeLLM | 6.94 | 16.25 | 33.19 | 50.28 |
| DeepSeek1.5B | STH-SepNet-Static | 6.79 | 16.37 | 30.26 | 48.81 |
| DeepSeek1.5B | STH-SepNet-GNN | 5.74 | 14.55 | 21.73 | 35.47 |
| DeepSeek1.5B | **STH-SepNet** | **5.27** | **14.19** | **21.39** | **34.96** |

---

# 4.7 Ablation Study

We conduct three ablation studies to validate the effectiveness of key components in the proposed framework, exploring:

1. The role of LLMs.
2. The mixed-order spatio-temporal convolutional networks.
3. The order of hypergraph on forecast results.

## 4.7.1 LLMs Play a Critical Role in Temporal Modeling

To assess the necessity of LLMs in spatio-temporal prediction and their synergistic effects with the adaptive hypergraph structure, we compare:

- STH-SepNet without LLMs
- STH-SepNet with BERT
- STH-SepNet with GPT
- STH-SepNet with LLAMA
- STH-SepNet with DeepSeek

The comparison reveals substantial performance improvements, particularly in capturing long-range dependencies and multi-scale periodicity.

![Figure 3: Performance comparison of MAE](figure-3-placeholder)

**Figure 3:** Performance comparison of MAE between STH-SepNet trained on different datasets.

## 4.7.2 Lightweight LLMs Achieve Competitive Performance

The results show that incorporating BERT demonstrates notable accuracy. On BIKE-Inflow, BERT not only surpasses larger backbones such as LLAMA7B but does so with fewer parameters:

- BERT: 110M parameters
- LLAMA7B: 6740M parameters

This indicates that excessive parameter scaling is not essential for effective temporal modeling.

Performance variations across different LLM backbones remain minimal, with fluctuations under 3% on BJ500. This stability arises from the decoupled nature of the adaptive hypergraph, which independently captures spatial dependencies, reducing reliance on LLM scale.

## 4.7.3 Effective Order of the Hypergraph

The proposed framework leverages KNN to construct hyperedges in hypergraphs, addressing limitations of spatial dependencies that traditional spatio-temporal convolutions are unable to model.

![Figure 4: Analysis of effective order](figure-4-placeholder)

**Figure 4:** Analysis of effective order on adaptive hypergraph.

The results show that the effective order $k=3$ significantly enhances model performance. When $k=2$, the high-order structure degenerates into a pairwise relationship. As $k$ increases, the model error initially decreases and then increases.

This phenomenon is attributed to:

- $k=2$: pairwise relationships fail to capture underlying dependencies.
- Larger $k$: higher-order structures may overfit coupled interactions.
- $k=3$: effectively characterizes evolving spatial dependencies.

## 4.7.4 Computational Efficiency Analysis

To analyze the advantages of decoupling in algorithmic efficiency, the STH-SepNet model is tested with multiple large language models by comparing GPU usage and training speed on different datasets using an NVIDIA A6000.

![Figure 5: GPU and time complexity](figure-5-placeholder)

**Figure 5:** Comparison of GPU and time complexity.

The results show that STH-SepNet series outperform TimeLLM in computational efficiency across most datasets.

For instance, STH-SepNet with BERT shows:

- GPU memory usage: 24.6G
- Training speed: 392 Epoch/s on BIKE-Inflow/Outflow

As the parameter size of LLMs increases, computational efficiency tends to decrease. However, larger model parameters do not necessarily enhance accuracy.

This advantage stems from STH-SepNet’s decoupled processing of temporal features and the use of average pooling to extract global trend features.

---

# 5. Conclusion

This paper introduces **STH-SepNet**, a framework for spatio-temporal prediction that decouples temporal and spatial modeling through two specialized components:

1. Lightweight large language models for temporal dynamics.
2. Adaptive hypergraphs for spatial dependencies.

By employing a spatio-temporal decoupling design, the ability of STH-SepNet to predict spatio-temporal data is significantly enhanced.

Experimental results demonstrate improved accuracy across diverse datasets, including:

- PEMS03: MAE 21.03 vs. 26.84 for non-LLM variants.
- BIKE-Outflow: MAE 5.33 vs. 6.74 for LLM baselines.

The adaptive hypergraph structure dynamically adjusts to spatial distribution shifts, such as policy-driven traffic pattern changes or sudden disruptions, enabling robust predictions in dynamic environments.

The improved performance is attributed to the decoupled architecture, which allows temporal and spatial modules to focus on distinct patterns without mutual interference. Adaptive hypergraphs address the limitations of static graph structures by modeling higher-order interactions and real-time spatial drift, while lightweight LLMs efficiently capture temporal trends.

## Limitations and Future Work

While STH-SepNet demonstrates strong performance, its current design has limitations.

The framework assumes temporal and spatial dependencies can be cleanly decoupled, which may not hold in scenarios where these dimensions are intrinsically intertwined, such as rapidly evolving events with coupled spatio-temporal causality.

Additionally, the adaptive hypergraph’s reliance on real-time node feature updates could pose challenges in latency-critical applications, where computational overhead for dynamic hyperedge generation might limit responsiveness.

In future work, the authors plan to explore hybrid architectures that balance decoupling and controlled interaction mechanisms.

---

# Acknowledgments

This work is supported by:

- National Key R&D Program of China under Grant No. 2022ZD0120004
- Zhishan Youth Scholar Program
- National Natural Science Foundation of China under Grant Nos. 62233004, 62273090
- Jiangsu Provincial Scientific Research Center of Applied Mathematics under Grant No. BK20233002

---

# Appendix A. Hypergraph Theory

## Definition A.1. Hyperedge

Given a high-order graph:

$$
H = (V,E)
$$

a hyperedge $e \in E$ is a non-empty subset of $V$.

For each $e \in E$:

$$
e \ne \emptyset
$$

and:

$$
e = (v_{i_1}, v_{i_2}, \ldots, v_{i_k}),
\quad
v_{i_j} \in V
$$

## Definition A.2. k-uniform Hyperedge

If a hyperedge $e \in E$ contains exactly $k$ vertices, then it is called a $k$-uniform hyperedge.

Formally, for each $e \in E$:

$$
|e| = k
$$

## Definition A.3. k-hops Neighborhoods

Given a node $v_i$, its $k$-hops neighborhood $N_k(v_i)$ comprises all nodes that can be reached from $v_i$ via at most $k$ edges from $v_i$.

## Proof for Theorem 1

The theorem is proved by showing both directions of the equivalence.

### Sufficiency

Assume:

$$
w \in N_{k-1}(v)
$$

We need to show that there exists a $k$-order hyperedge $e \in H_v^k$ such that $w \in e$ and $e$ contains $v$, $w$, and at most $k-2$ intermediate nodes.

By the local connectivity condition, there exists a path from $v$ to $w$ using at most $k-1$ hyperedges. Let this path be represented by the sequence:

$$
e_1, e_2, \ldots, e_{k-1}
$$

Since each hyperedge can connect more than two nodes, we can construct a $k$-order hyperedge $e$ that includes $v$ and $w$ along with at most $k-2$ intermediate nodes. This satisfies the hyperedge coverage condition.

If there are multiple such hyperedges, the uniqueness condition ensures that they share the same set of intermediate nodes, thus ensuring consistency.

### Necessity

Assume there exists a $k$-order hyperedge:

$$
e \in H_v^k
$$

such that $w \in e$, and $e$ contains $v$, $w$, and at most $k-2$ intermediate nodes.

We need to show that:

$$
w \in N_{k-1}(v)
$$

By definition, the hyperedge $e$ connects $v$ and $w$ through at most $k-2$ intermediate nodes. This implies that there is a path from $v$ to $w$ consisting of at most $k-1$ hyperedges. Thus, $w$ is within the $(k-1)$-hops neighborhood of $v$.

## Algorithm A.1. Hyperedge Construction

```text
Require: Batch data [B, L, N, F], high-order parameter k
Ensure: Constructed hyperedges and spatial interaction results

1: Initialize hyperedges set: hyperedges = ∅
2: for each node in V, where |V| = N, do
3:     Find k nearest neighbors using KNN
4:     Construct dynamic k-order hyperedge
5:     Add the hyperedge to hyperedges
6: end for
7: Construct adaptive hypergraph based on hyperedges set
```

---

# Appendix B. Experiment Settings and Results

## Table B.1. Comparison of Parameter Sizes and Dimensions Across LLMs

| Model | Parameters | LLM Dimension |
|---|---:|---:|
| BERT | 110M | 768 |
| GPT-2 | 124M | 768 |
| GPT-3 | 7580M | 4096 |
| LLAMA-1B | 1230M | 2048 |
| LLAMA-7B | 6740M | 4096 |
| LLAMA-8B | 8000M | 4096 |
| DeepSeek-Qwen1.5B | 1500M | 1536 |

## Dataset Details

### BIKE-Inflow / BIKE-Outflow

The dataset captures bicycle demand across 295 traffic nodes in New York, recorded hourly. The dataset spans from 2023-01-01 00:00 to 2024-01-01 23:00.

### PEMS03

The dataset contains traffic speed data from 358 stations in the California Highway System, with a 5-minute interval. The time range covers weekdays from 2008-01-01 00:00 to 2008-03-31 23:55:00.

### BJ500

The dataset consists of traffic speed information from 500 stations in the Beijing Highway System, also at 5-minute intervals. The dataset covers weekdays from 2020-07-01 00:00:00 to 2020-07-31 23:55:00.

### METR-LA

The dataset from the Los Angeles Metropolitan Transportation Authority contains average traffic speed measured by 207 loop detectors on the highways of Los Angeles County, ranging from March 2012 to June 2012.

## Prompt Example on PEMS03

```text
[Dataset Description]
Pems03 Traffic dataset: The data consists of 358 selected stop data distributed in the California Highway System (CalTrans). The data set time interval is a time interval of 5 minutes. The time range is in 2008-01-01 00:00:00 to 2008-03-31 23:55:00 weekdays of 358 stations traffic speed information. Data shows a strong periodicity. The input consists of adjacency matrix and timing feature matrix. The adjacency matrix is a new matrix with similar traffic flow patterns by analyzing the characteristics of existing spatio-temporal traffic data, and the characteristic matrix is the time series characteristic matrix of each sensor node.

[Task Instruction]
Forecast the next L steps given the previous H steps.

[Statistical Information]
The timestamp information is formatted as [month, day, hour, minutes]. The input time begins from <start time> to <end time>, and the prediction time spans from <start prediction time> to <end prediction time>. The minimum value is <min value>, the maximum value is <max value>, and the median value is <median value>. The trend of the input is either upward or downward. The top 5 lags are <lag values>.
```

---

# Appendix C. Ablation Experiment Results

## Ablation Study for RQ3 and RQ4

The authors compare two model variants:

- **STH-SepNet-w/o:** without large language models; retains only the adaptive hypergraph and linear temporal layers.
- **STH-SepNet-w/i:** full framework equipped with LLM backbones, including BERT, GPT, LLAMA, and DeepSeek.

On PEMS03, excluding LLMs results in significantly higher errors:

- Without LLMs: MAE 26.84, RMSE 43.44
- With BERT: MAE 21.03, RMSE 34.17

This corresponds to increases of 21.7% and 21.3%, respectively.

## Table C.1. Performance Comparison of Different LLMs

Prediction horizon: 48 time steps. Lookback window: $T=48$.

| Model | BIKE-Inflow MAE | BIKE-Inflow RMSE | BIKE-Outflow MAE | BIKE-Outflow RMSE | PEMS03 MAE | PEMS03 RMSE | BJ500 MAE | BJ500 RMSE | METR-LA MAE | METR-LA RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STH-SepNet-w/o LLMs | 5.47 | 14.01 | 5.83 | 15.17 | 26.84 | 43.44 | 6.24 | 10.81 | 10.31 | 18.26 |
| STH-SepNet BERT-w/i | 5.18 | 14.40 | 5.33 | 14.23 | 21.03 | 34.17 | 5.58 | 9.77 | 9.42 | 16.41 |
| STH-SepNet GPT2-w/i | 5.35 | 14.71 | 5.31 | 14.24 | 21.43 | 35.01 | 5.88 | 9.69 | 9.57 | 16.64 |
| STH-SepNet GPT3-w/i | 5.36 | 14.78 | 5.24 | 14.16 | 21.13 | 34.69 | 5.83 | 9.92 | 9.56 | 16.77 |
| STH-SepNet LLAMA1B-w/i | 5.36 | 14.80 | 5.29 | 14.20 | 21.37 | 34.92 | 6.03 | 10.08 | 9.64 | 16.84 |
| STH-SepNet LLAMA7B-w/i | 5.18 | 14.24 | 5.34 | 14.31 | 21.52 | 35.27 | 5.73 | 9.58 | 9.72 | 16.90 |
| STH-SepNet LLAMA8B-w/i | 5.30 | 14.66 | 5.28 | 14.20 | 21.51 | 35.19 | 5.80 | 9.65 | 9.71 | 16.94 |
| STH-SepNet DeepSeek-Qwen1.5B-w/i | 5.25 | 14.48 | 5.27 | 14.19 | 21.39 | 34.96 | 5.91 | 9.94 | 9.65 | 16.89 |

## Ablation Study for Gating Mechanism

The authors evaluate three fusion mechanisms across five datasets:

- Adaptive Gate
- LSTM Gate
- Attention Gate

![Figure C.1: Ablation studies of fusion mechanisms](figure-c1-placeholder)

**Figure C.1:** Ablation studies comparing three fusion mechanisms on various datasets. LLM backbone: BERT.

Adaptive gating achieves superior prediction accuracy on all benchmarks. In complex traffic scenarios like METR-LA and BJ500, it shows particularly significant performance advantages, with 17.7%–45.6% MAE reduction compared to cross-attention and LSTM-based gating mechanisms.

## Effective Order of Adaptive Hypergraph

Table C.2 illustrates that as the order of the adaptive hypergraph increases, the RMSE error of the model initially decreases and then increases.

Specifically, on Bike-Outflow and PEMS03, when $k=3$, STH-SepNet achieves the best RMSE values.

## Table C.2. Analysis of Effective Order $k \in \{2,3,4,5\}$

Metric: RMSE. LLM backbone: BERT.

| Dataset | Model | k=2 | k=3 | k=4 | k=5 |
|---|---|---:|---:|---:|---:|
| BIKE-Outflow | BERT | 13.36 | 14.23 | 14.89 | 15.37 |
| BIKE-Outflow | GPT2 | 14.48 | 14.24 | 15.29 | 15.58 |
| BIKE-Outflow | LLAMA1B | 14.80 | 14.20 | 15.54 | 16.19 |
| BIKE-Outflow | DeepSeek1.5B | 14.55 | 14.19 | 15.46 | 16.05 |
| PEMS03 | BERT | 34.52 | 34.17 | 35.81 | 37.84 |
| PEMS03 | GPT2 | 35.78 | 35.01 | 36.22 | 37.74 |
| PEMS03 | LLAMA1B | 35.87 | 34.92 | 37.16 | 39.07 |
| PEMS03 | DeepSeek1.5B | 35.47 | 34.96 | 38.03 | 39.27 |

## Ablation Study for Different Modules

The authors conduct ablation studies on the STH-SepNet model, which incorporates:

- Spatio-temporal modules
- Static graphs
- LLMs
- Adaptive graphs
- Hypergraph modules

![Figure C.2: Ablation studies of different modules](figure-c2-placeholder)

**Figure C.2:** Ablation studies for spatio, temporal, static graph, LLMs, adaptive graph, and hypergraph modules. LLM backbone: BERT.

The fully equipped model, integrating the adaptive hypergraph module, the spatio-temporal module, and BERT, demonstrates the most significant improvement in MAE and RMSE.

When the adaptive hypergraph module is removed and only the static graph module is retained, MAE and RMSE markedly increase, especially for BIKE-Inflow and BIKE-Outflow. This indicates that static graphs struggle to adequately depict dynamic and complex spatial associations, reflecting the irreplaceable role of the adaptive hypergraph module in learning dynamic spatial structures.

---

# References

1. Mohammad Taha Bahadori, Qi Rose Yu, and Yan Liu. 2014. Fast multivariate spatio-temporal analysis via low rank tensor learning.
2. Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. 2020. Adaptive graph convolutional recurrent network for traffic forecasting.
3. Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. 2023. Accurate medium-range global weather forecasting with 3D neural networks.
4. Ching Chang, Wen-Chih Peng, and Tien-Fu Chen. 2023. LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs.
5. Jeongwhan Choi, Hwangyong Choi, Jeehyun Hwang, and Noseong Park. 2022. Graph Neural Rough Differential Equations for Traffic Forecasting.
6. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding.
7. Zheng Fang, Qingqing Long, Guojie Song, and Kunqing Xie. 2021. Spatial-temporal graph ODE networks for traffic flow forecasting.
8. Haotian Gao, Renhe Jiang, Zheng Dong, Jinliang Deng, Yuxin Ma, and Xuan Song. 2024. Spatial-temporal-decoupled masked pre-training for spatiotemporal forecasting.
9. Adam Goodge, Wee Siong Ng, Bryan Hooi, and See Kiong Ng. 2025. Spatio-Temporal Foundation Models: Vision, Challenges, and Opportunities.
10. Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew G. Wilson. 2024. Large language models are zero-shot time series forecasters.
11. Daya Guo et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.
12. Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. 2019. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting.
13. Edward J. Hu et al. 2021. LoRA: Low-rank adaptation of large language models.
14. Jing Huang and Jie Yang. 2021. UniGNN: A Unified Framework for Graph and Hypergraph Neural Networks.
15. Ziyu Jia et al. 2021. Multi-view spatial-temporal graph convolutional networks with domain generalization for sleep stage classification.
16. Ming Jin et al. 2024. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.
17. Ming Jin et al. 2023. Time-LLM: Time series forecasting by reprogramming large language models.
18. Thomas N. Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks.
19. Fuxian Li et al. 2023. Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution.
20. Jiawei Li, Jingshu Peng, Haoyang Li, and Lei Chen. 2024. UniCL: A Universal Contrastive Learning Framework for Large Time Series Models.
21. Zhonghang Li et al. 2024. OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction.
22. Zhonghang Li, Lianghao Xia, Yong Xu, and Chao Huang. 2024. GPT-ST: Generative pre-training of spatio-temporal graph neural networks.
23. Hangchen Liu et al. 2023. STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformers SOTA for Traffic Forecasting.
24. Yong Liu et al. 2023. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
25. Tung Nguyen et al. 2023. ClimaX: A foundation model for weather and climate.
26. Tong Nie et al. 2024. ImputeFormer: Low rankness-induced transformers for generalizable spatiotemporal imputation.
27. Yuqi Nie et al. 2022. A time series is worth 64 words: Long-term forecasting with transformers.
28. Yuqi Nie et al. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
29. Alec Radford et al. 2019. Language models are unsupervised multitask learners.
30. Zongjiang Shang, Ling Chen, Binqing Wu, and Dongliang Cui. 2024. AdaMSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting.
31. Zezhi Shao et al. 2022. Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting.
32. Chao Song et al. 2020. Spatial-temporal synchronous graph convolutional networks.
33. Hugo Touvron et al. 2023. LLaMA: Open and efficient foundation language models.
34. Haixu Wu et al. 2022. TimesNet: Temporal 2D-variation modeling for general time series analysis.
35. Haixu Wu et al. 2023. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
36. Haixu Wu et al. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting.
37. Zonghan Wu et al. 2020. Connecting the dots: Multivariate time series forecasting with graph neural networks.
38. Hao Xue and Flora D. Salim. 2023. PromptCast: A new prompt-based learning paradigm for time series forecasting.
39. Xiaodong Yan et al. 2023. Spatio-temporal hypergraph learning for next POI recommendation.
40. An Yang et al. 2024. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement.
41. Bing Yu, Haoteng Yin, and Zhanxing Zhu. 2018. Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.
42. Yuan Yuan et al. 2024. UniST: A prompt-empowered universal model for urban spatio-temporal prediction.
43. Ailing Zeng et al. 2023. Are transformers effective for time series forecasting?
44. Chuanpan Zheng et al. 2020. GMAN: A graph multi-attention network for traffic prediction.
45. Haoyi Zhou et al. 2021. Informer: Beyond efficient transformer for long sequence time-series forecasting.
46. Tian Zhou et al. 2022. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting.
47. Tian Zhou et al. 2023. One fits all: Power general time series analysis by pretrained LM.