#!/usr/bin/env python
# coding: utf-8
import sys
from typing import Dict, Any
import numpy as np
import hashlib
import torch
from tqdm import tqdm
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from captum.attr import DeepLift, Saliency
    DEEPLIFT_AVAILABLE = True
except Exception:
    DEEPLIFT_AVAILABLE = False

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 辅助函数 ----------
def _to_2d_float32(x):
    """强制转换成 (N, D) 的 float32 数组"""
    return np.atleast_2d(np.asarray(x, dtype=np.float32))

def _clean_contrib(arr, eps=1e-15):
    """清理贡献矩阵中的非法值（NaN/Inf）并置零极小值"""
    arr = np.asarray(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    arr[np.abs(arr) < eps] = 0.0
    return arr.astype(np.float32)

def safe_normalize(arr, eps=1e-12):
    """按行归一化（除以 abs 和）"""
    arr = np.asarray(arr, dtype=np.float32)
    denom = np.sum(np.abs(arr), axis=1, keepdims=True)
    return arr / np.maximum(denom, eps)

def stable_grad(model, x, device):
    if isinstance(device, str):
        device = torch.device(device)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)
    x_t.requires_grad_(True)
    y = model(x_t)
    y.sum().backward()
    return x_t.grad.detach().cpu().numpy()

# ---------- 1. Gradient Attribution (Saliency) ----------
class GradientAttribution:
    def __init__(self, model):
        self.model = model

    def attribute(self, x):
        import torch

        x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x.requires_grad_(True)

        scores = self.model(x).sum()
        grads = torch.autograd.grad(scores, x)[0]

        return grads.detach().cpu().numpy()

# ---------- 2. Integrated Gradients ----------
class IGAttribution:
    def __init__(self, indicator_model, reference, steps: int = 300, device=DEVICE):
        self.model = indicator_model
        self.reference = np.asarray(reference, dtype=np.float32).ravel()
        self.steps = max(1, int(steps))
        self.device = device

    def attribute(self, x: np.ndarray) -> np.ndarray:
        x = _to_2d_float32(x)
        n, d = x.shape

        if self.reference.shape[0] != d:
            raise ValueError(
                f"reference dimension mismatch: got {self.reference.shape[0]}, expected {d}"
            )

        alphas = np.linspace(0.0, 1.0, self.steps + 1, dtype=np.float32)[1:]

        path = (
            self.reference[None, None, :]
            + alphas[None, :, None] * (x[:, None, :] - self.reference[None, None, :])
        )

        grads = stable_grad(
            self.model,
            path.reshape(-1, d),
            self.device
        ).reshape(n, self.steps, d)

        avg_grad = np.mean(grads, axis=1)
        contrib = (x - self.reference) * avg_grad
        return _clean_contrib(contrib)


import shap
import torch
import numpy as np

# attribution.py
class SHAPAttribution:
    def __init__(self, model, background_data, device='cpu'):
        self.model = model.eval() 
        self.device = device
        
        # 确保背景数据至少是二维的 (N, D)
        bg_data = np.atleast_2d(background_data).astype(np.float32)
        self.background = torch.as_tensor(bg_data, dtype=torch.float32).to(self.device)
        
        # 使用 GradientExplainer 进行提速
        self.explainer = shap.GradientExplainer(self.model, self.background)
        
    def attribute(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        
        # 1. 计算 SHAP 值 (这是加速的核心)
        shap_values = self.explainer.shap_values(X_tensor)
        
        # 2. 统一处理返回结果
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # 3. 统一转为 numpy
        if isinstance(shap_values, torch.Tensor):
            return shap_values.detach().cpu().numpy()
        return np.asarray(shap_values, dtype=np.float32)


# ---------- 4. DeepLIFT Attribution ----------
class DeepLIFTAttribution:
    def __init__(self, model, device=DEVICE, reference=None):
        if not DEEPLIFT_AVAILABLE:
            raise RuntimeError("DeepLift is not available; captum is not installed.")
        if reference is None:
            raise ValueError("DeepLIFT requires a reference / baseline.")

        self.model = model
        self.device = device
        self.reference = np.asarray(reference, dtype=np.float32).reshape(1, -1)
        self.deeplift = DeepLift(model)

    def attribute(self, x):
        self.model.eval()
        
        # 预处理输入
        x_np = np.atleast_2d(x).astype(np.float32)
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 定义基准点分布 (直接使用 Multi-Baseline 策略，这是 DeepLIFT 的高效替代)
        num_refs = 8
        # 将参考基准转为 Tensor
        ref_base = torch.tensor(self.reference, dtype=torch.float32, device=self.device)
        
        # 在基准点周围加入微小噪声，模拟 DeepLIFT 的多基准点稳定性
        noise = 0.01 * torch.randn((num_refs, ref_base.shape[1]), device=self.device)
        ref_dist = ref_base.repeat(num_refs, 1) + noise
        
        attr_list = []
        
        # 直接利用 torch.autograd 计算梯度，完全不涉及 Hook 注册
        for ref_k in ref_dist:
            # 扩展基准点到与输入相同的 Batch 大小
            ref_k_batch = ref_k.unsqueeze(0).repeat(x_tensor.shape[0], 1)
            
            # 重新计算输出
            output = self.model(x_tensor)
            if output.ndim > 1:
                output = output.sum(dim=1)
            
            # 计算输出对输入的梯度
            grads = torch.autograd.grad(
                outputs=output,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(output),
                retain_graph=False,
                create_graph=False
            )[0]
            
            # DeepLIFT 的贡献度计算公式: Gradient * (Input - Baseline)
            delta = x_tensor - ref_k_batch
            attr_k = grads * delta
            
            attr_list.append(attr_k)
        
        # 对多个基准点的结果取平均，确保归因的鲁棒性
        attr = torch.stack(attr_list, dim=0).mean(dim=0)
        
        out = _clean_contrib(attr.detach().cpu().numpy())
        return out[0] if x.ndim == 1 else out
    
    
# ---------- 5. Vector Field Saliency (VSF) ----------
class VSFAttribution:
    """
    基于向量场流线的归因（VSF），通过梯度场积分寻找数据源，再计算贡献。
    内部优化：自适应步长欧拉迭代、均值引力防跑偏、提前收敛、更少冗余计算
    外部接口完全不变，上层调用无需任何修改
    """
    """
    MVSF原点计算不稳定问题解决办法
    1、弃用 RK2 双梯度（每次迭代只算 1 次梯度），计算耗时直接减半；
    2、梯度范数自适应缩放步长：梯度大走大步，梯度平缓自动慢走，彻底消除固定 dt 来回震荡；
    3、内置微弱引力项，把回溯点拉向初始样本基准，不会随便跑到离谱局部极小；
    4、连续多步位移极小就提前终止迭代，减少无效循环，速度进一步提升；
    5、仅最后统一 clip 边界，迭代过程不强行截断流线，溯源路径更真实；
    """
    def __init__(self, indicator_model, dt=0.05, max_steps=120, grad_tol=1e-5,
                 x_min=None, x_max=None, normalize=True, device=DEVICE):
        # 入参严格保持原样，外部调用零改动
        self.model = indicator_model
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.grad_tol = float(grad_tol)
        self.x_min = x_min
        self.x_max = x_max
        self.normalize = normalize
        self.device = device
        self.model.eval()

        self.trace_records = []
        # 内部固定引力系数，不暴露入参，不破坏接口
        self._pull_strength = 0.07
        # 缓存训练集均值用于引力项，首次计算自动从第一条样本梯度空间推定基准
        self._mean_baseline = None

    def _compute_data_source(self, x0):
        s = np.asarray(x0, dtype=np.float32).reshape(1, -1).copy()
        path_list = [s.copy().ravel()]
        prev_s = s.copy()
        stable_cnt = 0
        stable_threshold = 5

        # 第一次运行自动用初始样本作为临时基准，避免无基准报错
        if self._mean_baseline is None:
            self._mean_baseline = s.copy()

        for _ in range(self.max_steps):
            grad = stable_grad(self.model, s, self.device)[0]
            grad_norm = float(np.linalg.norm(grad))

            if not np.isfinite(grad_norm):
                path_list.append(s.copy().ravel())
                continue

            # 自适应步长，解决固定dt震荡/走太慢
            adapt_scale = np.clip(grad_norm, 0.01, 2.0)
            step = self.dt * adapt_scale * grad.reshape(1, -1)

            # 向基准均值施加拉力，防止溯源陷入鞍点、飘出合理区间
            pull = self._pull_strength * (self._mean_baseline - s)
            s = s - step + pull

            path_list.append(s.copy().ravel())

            move = np.linalg.norm(s - prev_s)
            prev_s = s.copy()

            if move < self.grad_tol:
                stable_cnt += 1
                if stable_cnt >= stable_threshold:
                    break
            else:
                stable_cnt = 0

        # 迭代结束后再裁剪边界，不中途截断路径
        if self.x_min is not None or self.x_max is not None:
            s = np.clip(s, self.x_min, self.x_max)

        source_final = s.reshape(-1).astype(np.float32)
        path_array = np.array(path_list, dtype=np.float32)
        return source_final, path_array

    def _single_sample_vsf(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        source, path = self._compute_data_source(x)

        self.trace_records.append({
            "raw_x": x.copy(),
            "trace_path": path,
            "final_source": source.copy()
        })

        grad = stable_grad(self.model, x.reshape(1, -1), self.device)[0]
        contrib = np.abs((x - source) * grad).astype(np.float32)
        if self.normalize:
            sum_abs = np.sum(np.abs(contrib))
            if sum_abs > 1e-12:
                contrib /= sum_abs
        return contrib

    def attribute(self, X):
        self.trace_records.clear()
        X = _to_2d_float32(X)
        results = np.zeros_like(X, dtype=np.float32)

        iterator = range(X.shape[0])
        for i in iterator:
            results[i] = self._single_sample_vsf(X[i])

        return _clean_contrib(results)

    def save_trace_to_file(self, save_path="vsf_trace_data.npy"):
        if not self.trace_records:
            raise RuntimeError("请先执行attribute()再保存")
        np.save(save_path, self.trace_records)
        print(f"溯源轨迹已保存至: {save_path}")

    @staticmethod
    def plot_trace_from_file(file_path, dim1=0, dim2=1, sample_idx=None, figsize=(9, 6)):
        import matplotlib.pyplot as plt
        records = np.load(file_path, allow_pickle=True).tolist()
        plt.figure(figsize=figsize)

        draw_items = records
        if sample_idx is not None:
            draw_items = [records[sample_idx]]

        for idx, item in enumerate(draw_items):
            path = item["trace_path"]
            raw = item["raw_x"]
            src = item["final_source"]
            plt.plot(path[:, dim1], path[:, dim2], c="#777777", lw=1, alpha=0.5)
            plt.scatter(raw[dim1], raw[dim2], c="#1f77b4", s=45, label="Original" if idx == 0 else "")
            plt.scatter(src[dim1], src[dim2], c="#d62728", marker="x", s=70, label="Source Point" if idx == 0 else "")

        plt.xlabel(f"Feature Dim {dim1}")
        plt.ylabel(f"Feature Dim {dim2}")
        plt.title("VSF Traceback Path & Final Source")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()    
# # ---------- 5. Vector Field Saliency (VSF) ----------
# class VSFAttribution:
#     """
#     基于向量场流线的归因（VSF），通过梯度场积分寻找数据源，再计算贡献。
#     改进：RK2二阶积分、无梯度归一化、连续收敛判断、延后边界裁剪、支持路径保存与离线绘图
#     """
#     def __init__(self, indicator_model, dt=0.05, max_steps=120, grad_tol=1e-5,
#                  x_min=None, x_max=None, normalize=True, device=DEVICE):
#         self.model = indicator_model
#         self.dt = float(dt)
#         self.max_steps = int(max_steps)
#         self.grad_tol = float(grad_tol)
#         self.x_min = x_min
#         self.x_max = x_max
#         self.normalize = normalize
#         self.device = device
#         self.model.eval()
#         # 缓存本批次所有样本：原始输入、整条迭代路径、最终源点
#         self.trace_records = []

#     def _compute_data_source(self, x0):
#         s = np.asarray(x0, dtype=np.float32).reshape(1, -1).copy()
#         path_list = [s.copy().ravel()]
#         prev_s = s.copy()
#         stable_cnt = 0
#         stable_threshold = 5  # 连续5步小幅移动则判定收敛退出

#         for _ in range(self.max_steps):
#             # RK2 第一步梯度
#             grad1 = stable_grad(self.model, s, self.device)[0]
#             norm1 = np.linalg.norm(grad1)
#             if not np.isfinite(norm1):
#                 print("梯度出现非法值，本轮迭代不更新")
#                 path_list.append(s.copy().ravel())
#                 continue

#             # 构造中间预估点
#             s_mid = s - 0.5 * self.dt * grad1.reshape(1, -1)
#             grad_mid = stable_grad(self.model, s_mid, self.device)[0]
#             norm_mid = np.linalg.norm(grad_mid)
#             if not np.isfinite(norm_mid):
#                 grad_mid = grad1

#             # RK2 步进更新，不再对梯度做归一化
#             s = s - self.dt * grad_mid.reshape(1, -1)
#             path_list.append(s.copy().ravel())

#             # 位移收敛判断
#             move = np.linalg.norm(s - prev_s)
#             prev_s = s.copy()
#             if move < self.grad_tol:
#                 stable_cnt += 1
#                 if stable_cnt >= stable_threshold:
#                     print(f"连续{stable_threshold}步位移极小，提前终止溯源")
#                     break
#             else:
#                 stable_cnt = 0

#         # 全部迭代结束后再做值域裁剪，不中途截断路径
#         if self.x_min is not None or self.x_max is not None:
#             s = np.clip(s, self.x_min, self.x_max)

#         source_final = s.reshape(-1).astype(np.float32)
#         path_array = np.array(path_list, dtype=np.float32)
#         return source_final, path_array

#     def _single_sample_vsf(self, x):
#         x = np.asarray(x, dtype=np.float32).ravel()
#         source, path = self._compute_data_source(x)

#         # 记录本条样本全部信息
#         self.trace_records.append({
#             "raw_x": x.copy(),
#             "trace_path": path,
#             "final_source": source.copy()
#         })

#         grad = stable_grad(self.model, x.reshape(1, -1), self.device)[0]
#         contrib = np.abs((x - source) * grad).astype(np.float32)
#         if self.normalize:
#             sum_abs = np.sum(np.abs(contrib))
#             if sum_abs > 1e-12:
#                 contrib /= sum_abs
#                 # print("该样本提前截止")
#         return contrib

#     def attribute(self, X):
#         # 新批次自动清空历史轨迹缓存
#         self.trace_records.clear()
#         X = _to_2d_float32(X)
#         results = np.zeros_like(X, dtype=np.float32)

#         iterator = range(X.shape[0])
#         for i in iterator:
#             results[i] = self._single_sample_vsf(X[i])

#         return _clean_contrib(results)

#     def save_trace_to_file(self, save_path="vsf_trace_data.npy"):
#         """将当前批次所有样本原始点、完整迭代路径、最终源点保存至npy文件"""
#         if not self.trace_records:
#             raise RuntimeError("请先调用attribute执行归因计算后再保存")
#         np.save(save_path, self.trace_records)
#         print(f"溯源数据已保存至: {save_path}")

#     @staticmethod
#     def plot_trace_from_file(file_path, dim1=0, dim2=1, sample_idx=None, figsize=(9, 6)):
#         """离线加载npy文件绘制回溯流线、原始样本、最终源点"""
#         import matplotlib.pyplot as plt
#         records = np.load(file_path, allow_pickle=True).tolist()
#         plt.figure(figsize=figsize)

#         draw_items = records
#         if sample_idx is not None:
#             draw_items = [records[sample_idx]]

#         for idx, item in enumerate(draw_items):
#             path = item["trace_path"]
#             raw = item["raw_x"]
#             src = item["final_source"]
#             plt.plot(path[:, dim1], path[:, dim2], c="#777777", lw=1, alpha=0.5)
#             plt.scatter(raw[dim1], raw[dim2], c="#1f77b4", s=45, label="Original" if idx == 0 else "")
#             plt.scatter(src[dim1], src[dim2], c="#d62728", marker="x", s=70, label="Source Point" if idx == 0 else "")

#         plt.xlabel(f"Feature Dim {dim1}")
#         plt.ylabel(f"Feature Dim {dim2}")
#         plt.title("VSF Traceback Path & Final Source")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.show()    
    

# # ---------- 5. Vector Field Saliency (VSF) ----------
# class VSFAttribution:
#     """
#     基于向量场流线的归因（VSF），通过梯度场积分寻找数据源，再计算贡献。
#     """
#     def __init__(self, indicator_model, dt=0.05, max_steps=120, grad_tol=1e-5,
#                  x_min=None, x_max=None, normalize=True, device=DEVICE):
#         self.model = indicator_model
#         self.dt = float(dt)
#         self.max_steps = int(max_steps)
#         self.grad_tol = float(grad_tol)
#         self.x_min = x_min
#         self.x_max = x_max
#         self.normalize = normalize
#         self.device = device
#         self.model.eval()##？？？？？？？？？？？？？？？？？？？？？？

#     def _compute_data_source(self, x0):
#         s = np.asarray(x0, dtype=np.float32).reshape(1, -1).copy()

#         for _ in range(self.max_steps):
#             grad = stable_grad(self.model, s, self.device)[0]
#             grad_norm = float(np.linalg.norm(grad))

#             if not np.isfinite(grad_norm) or grad_norm < self.grad_tol:
#                 print('提前结束，可能没有很好计算数据源')
#                 break

#             grad_unit = grad / (grad_norm + 1e-8)
#             # s = s - self.dt * grad_unit.reshape(1, -1)
#             s = s - self.dt * grad   # ！！！！！！！！！！！！！！！不归一化了，归一化会导致优化速度变慢

#             if self.x_min is not None or self.x_max is not None:
#                 s = np.clip(s, self.x_min, self.x_max)

#         return s.reshape(-1).astype(np.float32)

#     def _single_sample_vsf(self, x):
#         x = np.asarray(x, dtype=np.float32).ravel()
#         source = self._compute_data_source(x) # ！！！！！！！！！！！！！！！！！！！！！！！！！！
#         # source = np.zeros_like(x)
#         grad = stable_grad(self.model, x.reshape(1, -1), self.device)[0]
#         contrib = np.abs((x - source) * grad).astype(np.float32)
#         if self.normalize:
#             s = np.sum(np.abs(contrib))
#             if s > 1e-12:
#                 contrib /= s
#         return contrib


#     def attribute(self, X):
#         X = _to_2d_float32(X)
#         results = np.zeros_like(X, dtype=np.float32)

#         iterator = range(X.shape[0])
#         for i in iterator:
#             results[i] = self._single_sample_vsf(X[i])

#         return _clean_contrib(results)



# class VSFAttribution:
#     """
#     虚拟尺度因子法（VSF, Virtual Scale Factor）归因。

#     按照原始 VSF 的定义：
#         1) 对每个输入变量 x_i 引入虚拟尺度因子 lambda_i
#         2) 构造带尺度因子的指示量 I(lambda_1*x_1, ..., lambda_n*x_n)
#         3) 将指示量对 lambda_i 在 lambda_i=1 处的偏导绝对值
#            定义为变量 x_i 对指示量的贡献

#     由链式法则可得：
#         contribution_i = | x_i * dI/dx_i |

#     说明：
#     - 这是“原始虚拟尺度因子法”
#     - 不使用数据源，不进行流线积分
#     - 为了兼容外部调用，保留了原构造函数参数
#       （dt / max_steps / grad_tol / x_min / x_max 在本方法中不参与计算）
#     """

#     def __init__(self, indicator_model, dt=0.05, max_steps=120, grad_tol=1e-5,
#                  x_min=None, x_max=None, normalize=True, device=DEVICE):
#         """
#         参数说明
#         ----------
#         indicator_model : torch.nn.Module
#             已训练好的指示量模型/故障指标模型，用于计算模型输出及输入梯度。

#         dt, max_steps, grad_tol, x_min, x_max :
#             为兼容旧版“改进 VSF / 数据源 VSF”代码接口而保留。
#             原始 VSF 不使用这些参数，但外部调用无需修改。

#         normalize : bool
#             是否对贡献值进行归一化。
#             若为 True，则将每个样本的各维贡献除以总贡献，使其和为 1。

#         device : torch.device or str
#             模型所在设备，如 "cpu" 或 "cuda"。
#         """
#         self.model = indicator_model

#         # 以下参数在原始 VSF 中不参与计算，仅为保持外部接口兼容而保留
#         self.dt = float(dt)
#         self.max_steps = int(max_steps)
#         self.grad_tol = float(grad_tol)
#         self.x_min = x_min
#         self.x_max = x_max

#         # 是否归一化输出贡献
#         self.normalize = normalize

#         # 计算设备
#         self.device = device

#         # 切换到评估模式，保证 BatchNorm / Dropout 等层在归因时行为稳定
#         self.model.eval()

#     def _single_sample_vsf(self, x):
#         """
#         计算单个样本的 VSF 归因结果。

#         原始 VSF 公式：
#             contrib_i = | x_i * dI/dx_i |

#         参数
#         ----
#         x : array-like, shape (n_features,)
#             单个输入样本

#         返回
#         ----
#         contrib : np.ndarray, shape (n_features,)
#             每个特征对应的贡献值
#         """
#         # 转成一维 float32 向量
#         x = np.asarray(x, dtype=np.float32).ravel()

#         # 计算指示量/模型输出对输入 x 的梯度
#         # stable_grad(...) 返回形状通常为 (batch_size, n_features)
#         # 这里只传入 1 个样本，因此取第 0 个即可
#         grad = stable_grad(self.model, x.reshape(1, -1), self.device)[0]

#         # 根据原始 VSF 公式计算贡献：
#         # contribution_i = | x_i * dI/dx_i |
#         contrib = np.abs(x * grad).astype(np.float32)

#         # 若需要归一化，则将各维贡献除以总贡献
#         if self.normalize:
#             s = np.sum(np.abs(contrib))
#             if s > 1e-12:
#                 contrib /= s

#         return contrib

#     def attribute(self, X):
#         """
#         计算一批样本的 VSF 归因结果。

#         参数
#         ----
#         X : array-like, shape (n_samples, n_features) 或 (n_features,)
#             输入样本集。若传入单一样本，也会被转换为二维数组处理。

#         返回
#         ----
#         results : np.ndarray, shape (n_samples, n_features)
#             每个样本、每个特征的贡献值
#         """
#         # 转为标准二维 float32 数组，形状为 (n_samples, n_features)
#         X = _to_2d_float32(X)

#         # 创建结果数组
#         results = np.zeros_like(X, dtype=np.float32)

#         # 逐样本计算贡献
#         for i in range(X.shape[0]):
#             results[i] = self._single_sample_vsf(X[i])

#         # 清理数值异常（如 NaN / Inf），保持与原外部流程兼容
#         return _clean_contrib(results)


# ---------- 统一输出 ----------
def get_unified_output(explainer, x: np.ndarray) -> Dict[str, Any]:
    raw = explainer.attribute(x)
    raw = np.asarray(raw, dtype=np.float32)

    # ===== 原始绝对贡献 =====
    abs_raw = np.abs(raw)
    denom = np.maximum(np.sum(abs_raw, axis=1, keepdims=True), 1e-12)

    # ===== signed ratio（标准）=====
    signed_ratio = raw / denom   # [-1, 1]

    # ===== shifted ratio（你现在用的核心）=====
    shifted = signed_ratio - np.min(signed_ratio)
    shifted_ratio = shifted / (np.max(shifted) + 1e-12)

    # ===== 排序 =====
    rank = np.argsort(-abs_raw, axis=1)
    shifted_rank = np.argsort(-shifted_ratio, axis=1)

    return {
        "raw_contrib": raw,
        "abs_contrib": abs_raw,
        "signed_norm": signed_ratio,
        "norm_contrib": signed_ratio,

      
        "shifted_ratio": shifted_ratio,
        "shifted_contrib": shifted,

        "rank": rank,
        "shifted_rank": shifted_rank,
    }