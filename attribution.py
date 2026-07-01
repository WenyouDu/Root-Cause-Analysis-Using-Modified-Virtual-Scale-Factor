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

# ---------- 3. SHAP Attribution (PermutationExplainer) ----------
class SHAPAttribution:
    """
    基于 SHAP PermutationExplainer 的归因。
    背景数据会自动压缩（kmeans），支持缓存。
    """
    def __init__(self, score_numpy, background,
                 nsamples="auto", kmeans=20, batch_size=16,
                 use_cache=True):
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not installed.")
        self.score_func = score_numpy
        self.batch_size = int(batch_size)
        self._cache = {} if use_cache else None
        self._pred_cache = {}
        background = _to_2d_float32(background)

        # 背景压缩
        if kmeans is not None and kmeans > 0 and len(background) > kmeans:
            kmeans_obj = shap.kmeans(background, kmeans)
            self.background = kmeans_obj.data if hasattr(kmeans_obj, 'data') else kmeans_obj
        else:
            self.background = background

        # 创建 PermutationExplainer
        self.explainer = shap.PermutationExplainer(self._predict, self.background)

        # 期望值
        try:
            ev = self.explainer.expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                ev = np.asarray(ev).ravel()[0]
            self.expected_value = float(ev)
        except AttributeError:
            self.expected_value = float(np.mean(self._predict(self.background)))

        # 兼容 auto 逻辑
        if str(nsamples).lower() == "auto":
            self.max_evals = None
        else:
            self.max_evals = int(nsamples)

    def _predict(self, X):
        X = _to_2d_float32(X)
        key = hashlib.md5(X.tobytes()).hexdigest()
        if key in self._pred_cache:
            return self._pred_cache[key]
        y = self.score_func(X).reshape(-1).astype(np.float32)
        self._pred_cache[key] = y
        return y

    def attribute(self, X):
        X = _to_2d_float32(X)
        n_total, n_features = X.shape
        if n_total == 0:
            return np.empty((0, n_features), dtype=np.float32)

        # 缓存检查
        cache_key = None  # 修复未定义 bug
        if self._cache is not None:
            cache_key = hashlib.md5(X.tobytes()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

        results = []
        for start in range(0, n_total, self.batch_size):
            end = min(start + self.batch_size, n_total)
            chunk = X[start:end]

            if self.max_evals is not None:
                explanation = self.explainer(chunk, max_evals=self.max_evals)
            else:
                explanation = self.explainer(chunk)

            sv = explanation.values
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv, dtype=np.float32)
            if sv.ndim == 3 and sv.shape[-1] == 1:
                sv = sv[..., 0]

            results.append(sv)


        shap_values = np.vstack(results)
        shap_values = np.where(np.isfinite(shap_values), shap_values, 0.0).astype(np.float32)

        if self._cache is not None and cache_key is not None:
            self._cache[cache_key] = shap_values
        return shap_values


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

        x_np = np.asarray(x, dtype=np.float32)
        single_input = x_np.ndim == 1
        if single_input:
            x_np = x_np.reshape(1, -1)

        if self.reference.shape[1] != x_np.shape[1]:
            raise ValueError(
                f"reference dimension mismatch: got {self.reference.shape[1]}, expected {x_np.shape[1]}"
            )

        baseline_np = np.repeat(self.reference, x_np.shape[0], axis=0)

        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        x_tensor.requires_grad_(True)

        ref_tensor = torch.tensor(baseline_np, dtype=torch.float32, device=self.device)

        try:
            # ===== 正常 DeepLIFT =====
            attr = self.deeplift.attribute(x_tensor, baselines=ref_tensor)

        except RuntimeError as e:
            msg = str(e)

            if (
                "does not contain some of the input/output attributes" in msg
                or "module is being used more than once" in msg
            ):
                print("[WARN] DeepLIFT structural failure → fallback (multi-baseline DeepLIFT-like)")

                # ===== multi-baseline fallback =====
                num_refs = 8

                noise = 0.01 * torch.randn((num_refs, ref_tensor.shape[1]), device=self.device)
                ref_dist = ref_tensor[:1].repeat(num_refs, 1) + noise

                attr_list = []

                for ref_k in ref_dist:
                    ref_k = ref_k.unsqueeze(0).repeat(x_tensor.shape[0], 1)

                    x_tensor.requires_grad_(True)

                    output = self.model(x_tensor)

                    if output.ndim > 1:
                        output = output.sum(dim=1)

                    grads = torch.autograd.grad(
                        outputs=output,
                        inputs=x_tensor,
                        grad_outputs=torch.ones_like(output),
                        retain_graph=False,
                        create_graph=False
                    )[0]

                    delta = x_tensor - ref_k
                    attr_k = grads * delta

                    attr_list.append(attr_k)

                attr = torch.stack(attr_list, dim=0).mean(dim=0)

            else:
                raise e

        out = _clean_contrib(attr.detach().cpu().numpy())
        return out[0] if single_input else out

# ---------- 5. Vector Field Saliency (VSF) ----------
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
#                 break

#             grad_unit = grad / (grad_norm + 1e-8)
#             s = s - self.dt * grad_unit.reshape(1, -1)

#             if self.x_min is not None or self.x_max is not None:
#                 s = np.clip(s, self.x_min, self.x_max)

#         return s.reshape(-1).astype(np.float32)

#     def _single_sample_vsf(self, x):
#         x = np.asarray(x, dtype=np.float32).ravel()
#         source = self._compute_data_source(x)
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
class VSFAttribution:
    """
    虚拟尺度因子法（VSF, Virtual Scale Factor）归因。

    按照原始 VSF 的定义：
        1) 对每个输入变量 x_i 引入虚拟尺度因子 lambda_i
        2) 构造带尺度因子的指示量 I(lambda_1*x_1, ..., lambda_n*x_n)
        3) 将指示量对 lambda_i 在 lambda_i=1 处的偏导绝对值
           定义为变量 x_i 对指示量的贡献

    由链式法则可得：
        contribution_i = | x_i * dI/dx_i |

    说明：
    - 这是“原始虚拟尺度因子法”
    - 不使用数据源，不进行流线积分
    - 为了兼容外部调用，保留了原构造函数参数
      （dt / max_steps / grad_tol / x_min / x_max 在本方法中不参与计算）
    """

    def __init__(self, indicator_model, dt=0.05, max_steps=120, grad_tol=1e-5,
                 x_min=None, x_max=None, normalize=True, device=DEVICE):
        """
        参数说明
        ----------
        indicator_model : torch.nn.Module
            已训练好的指示量模型/故障指标模型，用于计算模型输出及输入梯度。

        dt, max_steps, grad_tol, x_min, x_max :
            为兼容旧版“改进 VSF / 数据源 VSF”代码接口而保留。
            原始 VSF 不使用这些参数，但外部调用无需修改。

        normalize : bool
            是否对贡献值进行归一化。
            若为 True，则将每个样本的各维贡献除以总贡献，使其和为 1。

        device : torch.device or str
            模型所在设备，如 "cpu" 或 "cuda"。
        """
        self.model = indicator_model

        # 以下参数在原始 VSF 中不参与计算，仅为保持外部接口兼容而保留
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.grad_tol = float(grad_tol)
        self.x_min = x_min
        self.x_max = x_max

        # 是否归一化输出贡献
        self.normalize = normalize

        # 计算设备
        self.device = device

        # 切换到评估模式，保证 BatchNorm / Dropout 等层在归因时行为稳定
        self.model.eval()

    def _single_sample_vsf(self, x):
        """
        计算单个样本的 VSF 归因结果。

        原始 VSF 公式：
            contrib_i = | x_i * dI/dx_i |

        参数
        ----
        x : array-like, shape (n_features,)
            单个输入样本

        返回
        ----
        contrib : np.ndarray, shape (n_features,)
            每个特征对应的贡献值
        """
        # 转成一维 float32 向量
        x = np.asarray(x, dtype=np.float32).ravel()

        # 计算指示量/模型输出对输入 x 的梯度
        # stable_grad(...) 返回形状通常为 (batch_size, n_features)
        # 这里只传入 1 个样本，因此取第 0 个即可
        grad = stable_grad(self.model, x.reshape(1, -1), self.device)[0]

        # 根据原始 VSF 公式计算贡献：
        # contribution_i = | x_i * dI/dx_i |
        contrib = np.abs(x * grad).astype(np.float32)

        # 若需要归一化，则将各维贡献除以总贡献
        if self.normalize:
            s = np.sum(np.abs(contrib))
            if s > 1e-12:
                contrib /= s

        return contrib

    def attribute(self, X):
        """
        计算一批样本的 VSF 归因结果。

        参数
        ----
        X : array-like, shape (n_samples, n_features) 或 (n_features,)
            输入样本集。若传入单一样本，也会被转换为二维数组处理。

        返回
        ----
        results : np.ndarray, shape (n_samples, n_features)
            每个样本、每个特征的贡献值
        """
        # 转为标准二维 float32 数组，形状为 (n_samples, n_features)
        X = _to_2d_float32(X)

        # 创建结果数组
        results = np.zeros_like(X, dtype=np.float32)

        # 逐样本计算贡献
        for i in range(X.shape[0]):
            results[i] = self._single_sample_vsf(X[i])

        # 清理数值异常（如 NaN / Inf），保持与原外部流程兼容
        return _clean_contrib(results)


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