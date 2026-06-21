"""
train_sthsepnet.py — STH-SepNet 单卡训练，可选方案1/方案2

用法:
  # 原论文基线
  python train_sthsepnet.py --llm_model BERT --data inflow \
    --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295

  # 方案1: 软聚类池化
  python train_sthsepnet.py --llm_model BERT --data inflow \
    --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
    --use_cluster_pooling --num_clusters 8

  # 方案2: 神经超图生成器
  python train_sthsepnet.py --llm_model BERT --data inflow \
    --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
    --use_neural_hypergraph --scale_hyperedges 5

  # 两个方案一起用
  python train_sthsepnet.py --llm_model BERT --data inflow \
    --root_path ./dataset/Bike/ --data_path inflow.csv --node_num 295 \
    --use_cluster_pooling --use_neural_hypergraph
"""

import os, time, gc, random, argparse
import numpy as np
import torch, torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

SEED = 2021
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
gc.collect(); torch.cuda.empty_cache()

from models import ST_SepNet
from data_provider.data_factory import data_provider
from utils.tools import load_content


# ═══════════════════════════════════════════════════════════
# EarlyStopping
# ═══════════════════════════════════════════════════════════
class SimpleEarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience; self.counter = 0; self.best_score = None; self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path, "checkpoint"))
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ═══════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, device, args):
    total_mse, total_mae = [], []
    model.eval()
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -args.pred_len :, :].float()
        batch_y = batch_y[:, -args.pred_len :, :].to(device)

        pred_f = outputs.reshape(-1, outputs.size(-1)).cpu()
        true_f = batch_y.reshape(-1, batch_y.size(-1)).cpu()
        inv_p = torch.tensor(loader.dataset.inverse_transform(pred_f.numpy()), device=device)
        inv_t = torch.tensor(loader.dataset.inverse_transform(true_f.numpy()), device=device)
        inv_p = inv_p.reshape(outputs.shape)
        inv_t = inv_t.reshape(batch_y.shape)
        total_mse.append(mse_fn(inv_p, inv_t).item())
        total_mae.append(mae_fn(inv_p, inv_t).item())
    model.train()
    mse = np.mean(total_mse)
    return mse, np.sqrt(mse), np.mean(total_mae)  # MSE, RMSE, MAE


# ═══════════════════════════════════════════════════════════
# 方案应用
# ═══════════════════════════════════════════════════════════
def apply_solutions(model, args, device):
    """
    根据命令行标志在原模型上应用方案1/方案2。
    不修改任何源文件，仅在内存中替换。
    """
    info = []

    # ── 方案2: 神经超图生成器（替换 KNN）──
    if args.use_neural_hypergraph:
        from models.Solution2_NeuralHypergraphGenerator import NeuralHypergraphGenerator
        neural_hg = NeuralHypergraphGenerator(
            nnodes=args.node_num, node_dim=40, hidden_dim=128, max_k=args.scale_hyperedges,
        ).to(device)
        with torch.no_grad():
            neural_hg.emb.weight.data.copy_(model.sthgnn.hgc.emb1.weight.data)
            neural_hg.lin.weight.data.copy_(model.sthgnn.hgc.lin1.weight.data)
            neural_hg.lin.bias.data.copy_(model.sthgnn.hgc.lin1.bias.data)
        model.sthgnn.hgc = neural_hg
        info.append(f"Solution2: NeuralHypergraph (K={args.scale_hyperedges})")

    # ── 方案1: 软聚类池化（替换全局均值）──
    if args.use_cluster_pooling:
        from models.Solution1_ClusterPooling import SoftClusterPool, cluster_diversity_loss

        cluster_pool = SoftClusterPool(
            num_nodes=args.node_num, node_embed_dim=40,
            num_clusters=args.num_clusters, temperature=1.0, identity_dim=args.llm_dim,
        ).to(device)
        with torch.no_grad():
            node_emb = model.sthgnn.hgc.emb1.weight
            cluster_pool.cluster_centers.data = node_emb[:args.num_clusters].clone()

        # Monkey-patch: 绑定到模型供训练循环使用
        model._cluster_pool = cluster_pool
        model._cluster_diversity_loss = cluster_diversity_loss
        info.append(f"Solution1: ClusterPooling (C={args.num_clusters})")

    return " | ".join(info) if info else "baseline (original STH-SepNet)"


# ═══════════════════════════════════════════════════════════
# Cluster Pooling 的 LLM 路径（替代原 forecast 中的均值部分）
# ═══════════════════════════════════════════════════════════
def cluster_llm_forward(model, x_enc, x_mark_enc, x_mark_dec, device):
    """方案1: N→C→LLM→C→N 的完整路径"""
    B, T, N = x_enc.shape
    C = model._cluster_pool.num_clusters

    node_emb = model.sthgnn.hgc.emb1.weight
    x_clustered, soft_assign, cluster_ids = model._cluster_pool(x_enc, node_emb)

    # Prompt（基于 C 条曲线的统计量）
    min_v = torch.min(x_clustered, dim=1)[0]
    max_v = torch.max(x_clustered, dim=1)[0]
    med_v = torch.median(x_clustered, dim=1).values
    lags = model.calcute_lags(x_clustered.float())
    trends = x_clustered.float().diff(dim=1).sum(dim=1)

    prompt = []
    for b in range(B):
        prompt.append(
            f"<|start_prompt|>Dataset description: {model.description}"
            f"Task description: forecast the next {model.pred_len} steps; "
            f"min {min_v[b].tolist()[0]:.2f} max {max_v[b].tolist()[0]:.2f} "
            f"median {med_v[b].tolist()[0]:.2f} lags {lags[b].tolist()[:5]}<|<end_prompt|>|>"
        )
    prompt_ids = model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
    prompt_emb = model.llm_model.get_input_embeddings()(prompt_ids.to(device))
    source_emb = model.mapping_layer(model.word_embeddings.permute(1, 0)).permute(1, 0)

    x_pool = x_clustered.permute(0, 2, 1)  # (B, C, T)
    enc_out, n_vars = model.patch_embedding(x_pool.to(torch.bfloat16))
    # enc_out: (B*C, P, d_llm), n_vars = C

    # 注入聚类身份：reshape → add identity → reshape back
    P = enc_out.shape[1]  # number of patch tokens
    enc_out = enc_out.view(B, C, P, -1)  # (B, C, P, d_llm)
    enc_out = enc_out + cluster_ids.unsqueeze(0).unsqueeze(2)  # (1, C, 1, d_llm)
    enc_out = enc_out.view(B * C, P, -1)  # back to (B*C, P, d_llm)

    enc_out = model.reprogramming_layer(enc_out, source_emb, source_emb)  # (B*C, P, d_llm)

    # 展平 C 维度回 B，以 cat 到 prompt 后面
    enc_out = enc_out.view(B, C * P, -1)  # (B, C*P, d_llm)
    llama_enc = torch.cat([prompt_emb, enc_out], dim=1)  # (B, prompt_len + C*P, d_llm)
    dec_out = model.llm_model(inputs_embeds=llama_enc).last_hidden_state
    dec_out = dec_out[:, :, : model.d_ff]

    # 切掉 prompt 部分，只留 patch 输出
    prompt_len = prompt_emb.shape[1]
    dec_out = dec_out[:, prompt_len:, :]  # (B, C*P, d_ff)
    dec_out = dec_out.view(B, C, -1, model.d_ff)  # (B, C, tokens, d_ff)
    dec_out = dec_out.permute(0, 1, 3, 2)[:, :, :, -model.patch_nums :]  # (B, C, d_ff, patch_nums)
    dec_out = model.output_projection(dec_out)  # (B, C, pred_len)
    dec_out = dec_out.permute(0, 2, 1)  # (B, pred_len, C)

    # C → N
    dec_out_N = dec_out @ soft_assign.T  # (B, pred_len, N)
    return dec_out_N


# ═══════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    args.min_values = np.min(train_loader.dataset.data_x, 0)
    args.max_values = np.max(train_loader.dataset.data_x, 0)
    args.content = load_content(args)

    model = ST_SepNet.Model(args).float().to(device)
    solution_info = apply_solutions(model, args, device)
    print(f"Model: {args.llm_model} | Mode: {solution_info}")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=train_steps, pct_start=0.2, epochs=args.train_epochs, max_lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = SimpleEarlyStopping(patience=args.patience)

    setting = f"{args.model_id}_{args.llm_model}_{args.data}"
    if args.use_cluster_pooling:
        setting += f"_C{args.num_clusters}"
    if args.use_neural_hypergraph:
        setting += f"_K{args.scale_hyperedges}"
    path = os.path.join(args.checkpoints, f"{setting}-{args.model_comment}")

    use_cluster = args.use_cluster_pooling

    for epoch in range(args.train_epochs):
        model.train()
        epoch_start = time.time()
        train_losses = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

            if use_cluster:
                # 方案1: LLM 走聚类路径，STHGNN 走原路径
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    dec_out_N = cluster_llm_forward(model, batch_x, batch_x_mark, batch_y_mark, device)
                    x_enc_s = batch_x.unsqueeze(-1).to(torch.bfloat16)
                    sthgnn_enc = model.sthgnn(x_enc_s.transpose(1, 3)).squeeze(-1)
                    combined = torch.cat((dec_out_N, sthgnn_enc), dim=2)
                    gate = torch.sigmoid(model.gate_linear(combined))
                    outputs = gate * dec_out_N + (1 - gate) * sthgnn_enc
                    outputs = outputs.float()
                all_min = torch.tensor(model.min_value).squeeze().to(device)
                all_max = torch.tensor(model.max_value).squeeze().to(device)
                outputs = torch.clamp(outputs, min=all_min.view(1, 1, -1), max=all_max.view(1, 1, -1))
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -args.pred_len :, :].float()
            batch_y_s = batch_y[:, -args.pred_len :, :].to(device)
            loss = criterion(outputs, batch_y_s)

            if use_cluster:
                div_loss = model._cluster_diversity_loss(model._cluster_pool.cluster_centers)
                loss = loss + 0.1 * div_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        train_l = np.mean(train_losses)
        vali_mse, vali_rmse, vali_mae = validate(model, vali_loader, device, args)
        test_mse, test_rmse, test_mae = validate(model, test_loader, device, args)
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1:3d} | Train: {train_l:.4f} | "
              f"Vali  MSE:{vali_mse:.1f} RMSE:{vali_rmse:.2f} MAE:{vali_mae:.2f} | "
              f"Test  MSE:{test_mse:.1f} RMSE:{test_rmse:.2f} MAE:{test_mae:.2f} | "
              f"{time.time()-epoch_start:.0f}s")
        early_stopping(vali_mse, model, path)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Done. Checkpoint: {path}")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STH-SepNet training with optional solutions")

    # 数据
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--node_num", type=int, required=True)

    # 方案标志
    parser.add_argument("--use_cluster_pooling", action="store_true", default=False)
    parser.add_argument("--use_neural_hypergraph", action="store_true", default=False)
    parser.add_argument("--num_clusters", type=int, default=8)

    # 训练
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_comment", type=str, default="")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # 原模型参数（带默认值）
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="pool")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="learned")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--llm_layers", type=int, default=32)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fusion_gate", type=str, default="adaptive")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--scale_hyperedges", type=int, default=3)
    parser.add_argument("--gcn_true", type=bool, default=True)
    parser.add_argument("--hgcn_true", type=bool, default=False)
    parser.add_argument("--hgat_true", type=bool, default=True)
    parser.add_argument("--temporl_true", type=bool, default=True)
    parser.add_argument("--static", type=bool, default=False)
    parser.add_argument("--adaptive_hyperhgnn", type=str, default="hgat")
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--loader", type=str, default="modal")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--prompt_domain", type=int, default=0)
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--adjacency_path", type=str, default="adj.csv")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--is_training", type=int, default=1)

    args = parser.parse_args()
    args.enc_in = args.node_num
    args.dec_in = args.node_num
    args.c_out = args.node_num
    train(args)
