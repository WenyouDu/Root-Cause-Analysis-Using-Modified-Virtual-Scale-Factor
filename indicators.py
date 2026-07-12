import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.svm import OneClassSVM
from scipy.stats import f as f_dist
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass

@dataclass
class ExpConfig:
    n_normal: int = 6000

    # thresholds
    alpha_t2: float = 0.99
    alpha_ae: float = 0.99
    alpha_svdd: float = 0.99

    # AE
    ae_epochs: int = 60
    ae_lr: float = 1e-3
    ae_batch_size: int = 128
    ae_patience: int = 20
    ae_min_delta: float = 1e-5
    ae_hidden_dims: tuple = (8, 4)
    ae_latent_dim: int = 1
    ae_dropout: float = 0.1
    ae_noise_std: float = 0.05
    ae_weight_decay: float = 1e-4
    ae_latent_reg: float = 1e-3
    ae_latent_score_weight: float = 0.1
    ae_joint_weight_q: float = 0.5
    ae_joint_weight_t2: float = 0.5
    ae_score_mode: str = "joint"

    # gradient / explanation
    grad_batch_size: int = 256
    ig_steps: int = 40
    shap_bg_size: int = 32
    vsf_step = 0.05
    vsf_max_steps = 100
    vsf_grad_tol = 1e-5

    # SVDD
    svdd_degree: int = 2
    svdd_coef0: float = 1.0
    svdd_nu: float = 0.01
    svdd_kernel: str = "rbf"
    svdd_gamma: float = 0.3

    # plot
    contour_gridsize: int = 40
    fig_dpi: int = 180


CFG = ExpConfig()

SHAP_CACHE = {}
EXPLAIN_CACHE = {}
GRID_CACHE = {}

# --------------------------------------------
# 全局随机种子和设备
# --------------------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Autoencoder
# ============================================================
def _adaptive_hidden_dims(input_dim: int) -> Tuple[int, int]:
    h1 = max(8, min(48, input_dim // 2))
    h2 = max(4, min(24, input_dim // 4))
    return h1, h2


class RealAE(nn.Module):
    def __init__(self, input_dim: int, config=None):
        super().__init__()

        if config is None:
            raise ValueError("RealAE requires config.")

        cfg = config
        self.config = cfg
        self.input_dim = input_dim

        hidden_dims = cfg.ae_hidden_dims or _adaptive_hidden_dims(input_dim)
        latent_dim = getattr(cfg, "ae_latent_dim", None) or max(
            1,
            hidden_dims[-1] // 2 if hidden_dims else 4
        )

        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_mean = None
        self.latent_cov_inv = None
        self.q_mean = None
        self.q_std = None
        self.t2_mean = None
        self.t2_std = None

        self.joint_weight_q = float(getattr(cfg, "ae_joint_weight_q", 0.5))
        self.joint_weight_t2 = float(getattr(cfg, "ae_joint_weight_t2", 0.5))

        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error_tensor(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        return torch.mean((x - recon) ** 2, dim=1)

    @staticmethod
    def _mahalanobis_numpy(Z: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
        cov_inv = np.atleast_2d(np.asarray(cov_inv, dtype=np.float32))

        diff = Z - mu
        scores = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        return scores.astype(np.float32)

    def _fit_latent_statistics(self, x_train: np.ndarray):
        self.eval()

        with torch.no_grad():
            xt = torch.as_tensor(x_train, dtype=torch.float32, device=DEVICE)
            z = self.encode_tensor(xt).detach().cpu().numpy()
            if z.ndim == 1:
                z = z.reshape(-1, 1)

            q = self.reconstruction_error_tensor(xt).detach().cpu().numpy()

        mu_z = z.mean(axis=0).astype(np.float32)
        cov_z = np.cov(z, rowvar=False)
        cov_z = np.atleast_2d(cov_z).astype(np.float32)
        cov_z = cov_z + 1e-8 * np.eye(cov_z.shape[0], dtype=np.float32)
        cov_inv = np.linalg.pinv(cov_z).astype(np.float32)

        t2 = self._mahalanobis_numpy(z, mu_z, cov_inv)

        eps = 1e-12
        self.latent_mean = mu_z
        self.latent_cov_inv = cov_inv

        self.q_mean = float(np.mean(q))
        self.q_std = float(np.std(q) + eps)

        self.t2_mean = float(np.mean(t2))
        self.t2_std = float(np.std(t2) + eps)

    def latent_t2_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None or self.latent_cov_inv is None:
            raise RuntimeError("AE latent statistics are not fitted. Call fit() first.")

        x = x.to(DEVICE)
        z = self.encode_tensor(x)

        mu = torch.as_tensor(
            self.latent_mean,
            dtype=torch.float32,
            device=x.device
        ).view(1, -1)

        cov_inv = torch.as_tensor(
            self.latent_cov_inv,
            dtype=torch.float32,
            device=x.device
        )

        if z.ndim == 1:
            z = z.view(-1, 1)

        diff = z - mu
        t2 = torch.sum((diff @ cov_inv) * diff, dim=1)
        return t2

    def standardized_reconstruction_error_tensor(self, x: torch.Tensor) -> torch.Tensor:
        q = self.reconstruction_error_tensor(x)
        return (q - self.q_mean) / self.q_std

    def standardized_latent_t2_tensor(self, x: torch.Tensor) -> torch.Tensor:
        t2 = self.latent_t2_tensor(x)
        return (t2 - self.t2_mean) / self.t2_std

    def anomaly_score_tensor(self, x: torch.Tensor, mode: str = None) -> torch.Tensor:
        if self.latent_mean is None or self.latent_cov_inv is None:
            raise RuntimeError("AE latent statistics are not fitted. Call fit() first.")

        if mode is None:
            mode = getattr(self.config, "ae_score_mode", "joint")

        mode = str(mode).lower()

        if mode == "recon":
            score = self.reconstruction_error_tensor(x)

        elif mode == "latent":
            score = self.latent_t2_tensor(x)

        elif mode == "joint":
            q_std = self.standardized_reconstruction_error_tensor(x)
            t2_std = self.standardized_latent_t2_tensor(x)
            score = self.joint_weight_q * q_std + self.joint_weight_t2 * t2_std

        else:
            raise ValueError(f"Unknown AE score mode: {mode}")

        return score.unsqueeze(1)

    def score_samples(self, x: np.ndarray, mode: str = None) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)

        self.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
            score = self.anomaly_score_tensor(xt, mode=mode).squeeze(1)

        return score.detach().cpu().numpy().astype(np.float32)

    def fit(self, x_train: np.ndarray, verbose: bool = True) -> "RealAE":
        cfg = self.config
        x_train = np.asarray(x_train, dtype=np.float32)

        dataset = TensorDataset(torch.as_tensor(x_train, device=DEVICE))
        loader = DataLoader(
            dataset,
            batch_size=cfg.ae_batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.ae_lr,
            weight_decay=getattr(cfg, "ae_weight_decay", 1e-5)
        )

        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.train()

        for epoch in range(cfg.ae_epochs):
            epoch_loss = 0.0

            for (xb,) in loader:
                optimizer.zero_grad()
                recon = self(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(x_train)

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"AE Epoch {epoch + 1}/{cfg.ae_epochs}, "
                    f"MSE Loss: {epoch_loss:.6f}"
                )

            if epoch_loss < best_loss - cfg.ae_min_delta:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= cfg.ae_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        self._fit_latent_statistics(x_train)
        return self


# ============================================================
# SVDD
# 全正 + 连续 + 分位数阈值更稳定
# ============================================================
class SVDDModel:
    def __init__(
        self,
        nu=0.15,
        gamma=0.01,
        kernel="poly",
        degree=2,
        coef0=1.0
    ):
        self.nu = float(nu)
        self.gamma = float(gamma)
        self.kernel = kernel
        self.degree = int(degree)
        self.coef0 = float(coef0)

        self.model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            nu=self.nu
        )

        self.sv_ = None
        self.coef_ = None
        self.rho_ = None

        # 用训练正常样本 raw score 的最小值做平移
        self.score_shift_ = 0.0

    def fit(self, X, verbose=False):
        X = np.asarray(X, dtype=np.float32)

        self.model.fit(X)

        self.sv_ = self.model.support_vectors_.astype(np.float32)
        self.coef_ = self.model.dual_coef_.ravel().astype(np.float32)

        if hasattr(self.model, "offset_"):
            self.rho_ = float(self.model.offset_[0])
        else:
            self.rho_ = float(-self.model.intercept_[0])

        raw_train_scores = self.raw_score_samples(X)
        self.score_shift_ = float(np.min(raw_train_scores))

        if verbose:
            print(
                f"SVDD fitted "
                f"(kernel={self.kernel}, "
                f"degree={self.degree}, "
                f"nu={self.nu}, "
                f"n_sv={len(self.sv_)}, "
                f"score_shift={self.score_shift_:.6f})"
            )

        return self

    def _kernel(self, X):
        if self.kernel == "rbf":
            return rbf_kernel(
                X,
                self.sv_,
                gamma=self.gamma
            ).astype(np.float32)

        elif self.kernel == "poly":
            return polynomial_kernel(
                X,
                self.sv_,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0
            ).astype(np.float32)

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def raw_score_samples(self, X):
        X = np.asarray(X, dtype=np.float32)

        K_sv_x = self._kernel(X)
        decision = K_sv_x @ self.coef_ - self.rho_
        raw_score = -decision

        return raw_score.astype(np.float32)

    def score_samples(self, X):
        raw_score = self.raw_score_samples(X)
        score = raw_score - self.score_shift_
        return score.astype(np.float32)


# ============================================================
# Unified indicator model
# ============================================================
class UnifiedIndicatorModel(nn.Module):
    def __init__(
        self,
        method: str,
        model_params: Dict[str, Any],
        cfg=None
    ):
        super().__init__()

        if cfg is None:
            raise ValueError("UnifiedIndicatorModel requires cfg.")

        self.method = method
        self.cfg = cfg

        if method in ("T2", "T²"):
            self.register_buffer(
                "mean",
                torch.as_tensor(model_params["mean"], dtype=torch.float32)
            )
            self.register_buffer(
                "comp",
                torch.as_tensor(model_params["comp"], dtype=torch.float32)
            )
            self.register_buffer(
                "var",
                torch.as_tensor(model_params["var"], dtype=torch.float32)
            )

        elif method == "SVDD":
            self.register_buffer(
                "sv",
                torch.as_tensor(model_params["sv"], dtype=torch.float32)
            )
            self.register_buffer(
                "coef",
                torch.as_tensor(model_params["coef"], dtype=torch.float32)
            )

            self.rho = float(model_params["rho"])
            self.gamma = float(model_params["gamma"])
            self.kernel = model_params.get("kernel", "poly")
            self.degree = int(model_params.get("degree", 2))
            self.coef0 = float(model_params.get("coef0", 1.0))
            self.score_shift = float(model_params.get("score_shift", 0.0))

        elif method == "AE":
            self.ae_model = model_params["ae_model"]

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method in ("T2", "T²"):
            xc = x - self.mean
            scores = xc @ self.comp.T

            return torch.sum(
                (scores ** 2) / (self.var + 1e-12),
                dim=1,
                keepdim=True
            )

        if self.method == "SVDD":
            if self.kernel == "rbf":
                x2 = torch.sum(x ** 2, dim=1, keepdim=True)
                sv2 = torch.sum(self.sv ** 2, dim=1).unsqueeze(0)
                dist2 = x2 + sv2 - 2.0 * (x @ self.sv.T)
                K_x_sv = torch.exp(-self.gamma * dist2)

            elif self.kernel == "poly":
                K_x_sv = (
                    self.gamma * (x @ self.sv.T)
                    + self.coef0
                ) ** self.degree

            else:
                raise ValueError(f"Unsupported kernel: {self.kernel}")

            decision = K_x_sv @ self.coef - self.rho
            raw_score = -decision
            score = raw_score - self.score_shift

            return score.unsqueeze(1)

        if self.method == "AE":
            mode = getattr(self.cfg, "ae_score_mode", "joint")
            return self.ae_model.anomaly_score_tensor(x, mode=mode)

        raise ValueError(f"Unknown method: {self.method}")


# ============================================================
# Build indicators
# ============================================================
def build_models(
    x_normal: np.ndarray,
    config=None
) -> Dict[str, Any]:
    if config is None:
        raise ValueError("build_models requires config.")

    cfg = config
    x_normal = np.asarray(x_normal, dtype=np.float32)

    n_samples, n_features = x_normal.shape


# ---------------- T² ----------------
    from sklearn.decomposition import PCA

    n_features = x_normal.shape[1]

    if n_features > 2:
        # ==============================
        # 高维数据：PCA降维
        # ==============================
        pca_full = PCA().fit(x_normal)

        cum_var = np.cumsum(
            pca_full.explained_variance_ratio_
        )

        n_comp = np.searchsorted(
            cum_var,
            0.9
        ) + 1

        pca = PCA(
            n_components=n_comp,
            random_state=SEED
        ).fit(x_normal)

        x_normal_pca = pca.transform(x_normal)

    else:
        # ==============================
        # 低维数据：保持原始维度
        # ==============================
        pca = None
        n_comp = n_features

        x_normal_pca = x_normal.copy()

    
    print(
    "T² principal components:",
    n_comp
    )

    print(
        "Cumulative explained variance:",
        np.sum(
            pca.explained_variance_ratio_
        )
    )


    explained_var = pca.explained_variance_.astype(np.float32)
    explained_var_safe = np.maximum(explained_var, 1e-12)

    def t2_score_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        scores = pca.transform(x)
        return np.sum((scores ** 2) / explained_var_safe, axis=1)

    t2_meta = {
        "mean": pca.mean_.astype(np.float32),
        "comp": pca.components_.astype(np.float32),
        "var": explained_var_safe.astype(np.float32),
        "p": len(explained_var),
        "n": n_samples,
        "pca": pca,
    }

    t2_indicator = UnifiedIndicatorModel("T²", t2_meta, cfg).to(DEVICE)

    # ---------------- SVDD ----------------
    svdd = SVDDModel(
        nu=cfg.svdd_nu,
        gamma=cfg.svdd_gamma,
        kernel=cfg.svdd_kernel,
        degree=cfg.svdd_degree,
        coef0=cfg.svdd_coef0
    )

    svdd.fit(x_normal, verbose=False)

    def svdd_score_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return svdd.score_samples(x)

    def svdd_raw_score_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return svdd.raw_score_samples(x)

    svdd_meta = {
        "sv": svdd.sv_.astype(np.float32),
        "coef": svdd.coef_.astype(np.float32),
        "rho": float(svdd.rho_),
        "gamma": float(cfg.svdd_gamma),
        "kernel": cfg.svdd_kernel,
        "degree": int(cfg.svdd_degree),
        "coef0": float(cfg.svdd_coef0),
        "score_shift": float(svdd.score_shift_),
        "R2": 0.0,
        "sk_model": svdd,
    }

    svdd_indicator = UnifiedIndicatorModel("SVDD", svdd_meta, cfg).to(DEVICE)

    # ---------------- AE ----------------
    ae = RealAE(input_dim=n_features, config=cfg).fit(x_normal)

    def ae_score_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        mode = getattr(cfg, "ae_score_mode", "joint")
        return ae.score_samples(x, mode=mode)

    def ae_score_recon_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return ae.score_samples(x, mode="recon")

    def ae_score_latent_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return ae.score_samples(x, mode="latent")

    def ae_score_joint_numpy(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return ae.score_samples(x, mode="joint")

    ae_indicator = UnifiedIndicatorModel(
        "AE",
        {"ae_model": ae},
        cfg
    ).to(DEVICE)

    ae_meta = {
        "ae": ae,
        "device": DEVICE,
        "latent_dim": ae.latent_dim,
        "hidden_dims": ae.hidden_dims,
        "latent_mean": ae.latent_mean,
        "latent_cov_inv": ae.latent_cov_inv,
        "q_mean": ae.q_mean,
        "q_std": ae.q_std,
        "t2_mean": ae.t2_mean,
        "t2_std": ae.t2_std,
        "joint_weight_q": ae.joint_weight_q,
        "joint_weight_t2": ae.joint_weight_t2,
        "default_score_mode": getattr(cfg, "ae_score_mode", "joint"),
    }

    return {
        "T²": {
            "score_numpy": t2_score_numpy,
            "indicator_model": t2_indicator,
            "meta": t2_meta,
        },
        "SVDD": {
            "score_numpy": svdd_score_numpy,
            "raw_score_numpy": svdd_raw_score_numpy,
            "indicator_model": svdd_indicator,
            "meta": svdd_meta,
        },
        "AE": {
            "score_numpy": ae_score_numpy,
            "score_recon_numpy": ae_score_recon_numpy,
            "score_latent_numpy": ae_score_latent_numpy,
            "score_joint_numpy": ae_score_joint_numpy,
            "indicator_model": ae_indicator,
            "meta": ae_meta,
        },
    }


# ============================================================
# Thresholds and references
# ============================================================
def compute_thresholds(
    models: Dict[str, Any],
    x_normal: np.ndarray,
    config=None,
) -> Dict[str, float]:
    if config is None:
        raise ValueError("compute_thresholds requires config.")

    cfg = config
    x_normal = np.asarray(x_normal, dtype=np.float32)
    thresholds: Dict[str, float] = {}

    p = models["T²"]["meta"]["p"]
    n = models["T²"]["meta"]["n"]

    if n <= p:
        raise ValueError(f"T² threshold requires n > p, got n={n}, p={p}.")

    thresholds["T²"] = (
        p * (n - 1) * (n + 1) / (n * (n - p))
    ) * f_dist.ppf(cfg.alpha_t2, p, n - p)

    svdd_scores = models["SVDD"]["score_numpy"](x_normal)
    thresholds["SVDD"] = float(np.quantile(svdd_scores, cfg.alpha_svdd))

    ae_scores = models["AE"]["score_numpy"](x_normal)
    thresholds["AE"] = float(np.quantile(ae_scores, cfg.alpha_ae))

    return thresholds


def get_reference_points(x_normal: np.ndarray) -> Dict[str, np.ndarray]:
    ref = np.mean(x_normal, axis=0).astype(np.float32)

    return {
        "T²": ref,
        "SVDD": ref,
        "AE": ref,
    }


# ============================================================
# Gradient utility
# ============================================================
def compute_gradient_batch(
    model: UnifiedIndicatorModel,
    x: np.ndarray,
    batch_size: int = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if batch_size is None:
        batch_size = getattr(model.cfg, "grad_batch_size", 256)

    model.eval()
    grads = []

    for start in range(0, len(x), batch_size):
        chunk = x[start:start + batch_size]

        xi = torch.as_tensor(
            chunk,
            dtype=torch.float32,
            device=DEVICE
        ).requires_grad_(True)

        y = model(xi)

        grad = torch.autograd.grad(
            outputs=y,
            inputs=xi,
            grad_outputs=torch.ones_like(y),
            retain_graph=False,
            create_graph=False,
        )[0]

        grads.append(grad.detach().cpu().numpy())

    if len(grads) == 0:
        return np.zeros_like(x, dtype=np.float32)

    return np.vstack(grads).astype(np.float32)


# ============================================================
# Attribution base and unified output
# ============================================================
class BaseAttribution(ABC):
    @abstractmethod
    def attribute(self, x: np.ndarray) -> np.ndarray:
        ...


def get_unified_output(
    explainer: BaseAttribution,
    x: np.ndarray
) -> Dict[str, Any]:

    raw = explainer.attribute(x)
    raw = np.asarray(raw, dtype=np.float32)

    abs_raw = np.abs(raw)

    abs_denom = np.maximum(
        np.sum(abs_raw, axis=1, keepdims=True),
        1e-12
    )

    signed_norm = raw / abs_denom

    shifted = raw - np.min(raw, axis=1, keepdims=True)

    shifted_denom = np.maximum(
        np.sum(shifted, axis=1, keepdims=True),
        1e-12
    )

    shifted_ratio = shifted / shifted_denom

    rank = np.argsort(-abs_raw, axis=1)
    shifted_rank = np.argsort(-shifted_ratio, axis=1)

    return {
        "raw_contrib": raw,
        "abs_contrib": abs_raw,
        "signed_norm": signed_norm,
        "norm_contrib": signed_norm,
        "shifted_contrib": shifted,
        "shifted_ratio": shifted_ratio,
        "rank": rank,
        "shifted_rank": shifted_rank,
    }
