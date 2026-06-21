# STH-SepNet：解耦时空预测——轻量大模型遇上自适应超图

> 陈嘉文, 邵琦, 陈都鑫\*, 虞文武\*
>
> *KDD '25*, 2025年8月 · 代码: github.com/SEU-WENJIA/ST-SepNetLightweight-LLMs-Meet-Adaptive-Hypergraphs

---

## 问题逻辑链

```
根本矛盾：时空预测需同时捕捉时间+空间→耦合建模参数暴增+特征互相干扰
  ↓
洞察1：时间动态低秩(早晚高峰+周末少数因子驱动)→轻量LLM够用，67亿参数浪费
洞察2：空间依赖是动态高阶交互(空间漂移)→静态图捕捉不了
  ↓
解法：时空解耦——轻量BERT管全局时间节奏+自适应超图管动态空间+门控逐位置融合
```

---

## 核心概念

**低秩**：N条传感器曲线表面独立，实际少少数隐藏因子驱动→秩≪N。**费曼**：100人乐队都跟总谱→极少因子。

**空间漂移**：早高峰A→B强，晚高峰B→A强。静态图权重永远不变。

**自适应超图**：超边≥3节点。KNN每次前向传播重建——"此刻特征最近的k个节点"→超边。k=3最优。**费曼**：微信群。两人=边，"室友群"(4人)=超边在。三个路口同时堵→超边同捆→协同恶化显式建模。

---

## Method

**重编程**(继承TIMELLM)：时序补丁做Q，LLM文本原型做K/V→交叉注意力→"语言化"时序。

**Prompt-as-Prefix**：数据集描述+任务指令+统计特征前缀。

**门控融合**：`Gate=σ(FFN([H_LLM,H_Spatial]))`→`LLM⊙Gate+Spatial⊙(1-Gate)`。逐节点逐时间步独立权重。

---

## 开放问题

> "The framework assumes temporal and spatial dependencies can be cleanly decoupled, which may not hold in scenarios where these dimensions are intrinsically intertwined (e.g., rapidly evolving events with coupled spatio-temporal causality). Additionally, the adaptive hypergraph's reliance on real-time node feature updates could pose challenges in latency-critical applications."

> "In future work, we will tackle these constraints by delving into hybrid architectures that strike a balance between decoupling and controlled interaction mechanisms."（Section 5）

三条方向：①时空强耦合时解耦可能不成立；②延迟敏感场景超边生成是瓶颈；③混合架构——解耦+受控交互。

---

## 概念速查

| 概念 | 费曼 |
|------|------|
| **低秩** | 100人乐队跟总谱走 |
| **空间漂移** | 早高峰A→B，晚高峰B→A |
| **自适应超图** | KNN实时聚→今天拥堵团和昨天不一样 |
| **重编程** | LLM戴眼镜→时序"看起来"像词 |
| **PaP** | 解题前告诉LLM"这是交通数据、早晚高峰" |
| **门控融合** | 逐位置两个专家投票——暴雨灾区超图说话 |
| **k=3** | k=2退化成对无增益；k≥4过拟合 |
