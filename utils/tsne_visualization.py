from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D


def _extract_features(lightning_model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return feature embeddings for a batch from different module types."""
    if hasattr(lightning_model, "get_features_and_logits"):
        features, _ = lightning_model.get_features_and_logits(x)
        return features

    if hasattr(lightning_model, "model"):
        inner = lightning_model.model
        if hasattr(inner, "get_features"):
            return inner.get_features(x)
        if hasattr(inner, "get_features_and_logits"):
            features, _ = inner.get_features_and_logits(x)
            return features

    return lightning_model(x)


def plot_tsne_from_test_dataloader(
    lightning_model: torch.nn.Module,
    test_dataloader,
    save_path,
    class_names=None,
    max_samples: int = 2000,
    random_state: int = 42,
    class_palette=None,
):
    """Create and save a t-SNE figure from test-set feature embeddings."""
    features_all, labels_all = [], []

    was_training = lightning_model.training
    lightning_model.eval()

    try:
        device = next(lightning_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for batch in test_dataloader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            x = x.to(device)

            features = _extract_features(lightning_model, x)
            features_all.append(features.detach().cpu())
            labels_all.append(y.detach().cpu())

    if was_training:
        lightning_model.train()

    if not features_all:
        return False, "No features collected from test dataloader"

    X = torch.cat(features_all, dim=0).numpy()
    y = torch.cat(labels_all, dim=0).numpy()

    n_samples = X.shape[0]
    if n_samples < 3:
        return False, f"Not enough samples for t-SNE: {n_samples}"

    if max_samples is not None and n_samples > max_samples:
        rng = np.random.default_rng(random_state)
        keep_idx = rng.choice(n_samples, size=max_samples, replace=False)
        X = X[keep_idx]
        y = y[keep_idx]
        n_samples = X.shape[0]

    perplexity = min(30, max(5, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)

    if perplexity < 2:
        return False, f"Invalid t-SNE perplexity {perplexity} for {n_samples} samples"

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    X_2d = reducer.fit_transform(X)

    classes = np.unique(y)
    if class_palette is None and class_names is not None and len(class_names) == 4:
        # Fixed 4-class palette: red, green, yellow, blue.
        class_palette = ["#E53935", "#43A047", "#FDD835", "#1E88E5"]
    cmap = plt.get_cmap("tab10", max(len(classes), 1))

    plt.figure(figsize=(7.5, 6.0))
    for i, cls in enumerate(classes):
        idx = y == cls
        label = str(int(cls))
        if class_names is not None and int(cls) < len(class_names):
            label = class_names[int(cls)]
        if class_palette is not None and int(cls) < len(class_palette):
            color = class_palette[int(cls)]
        else:
            color = cmap(i)
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=22,
            alpha=0.92,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            label=label,
        )

    plt.title("t-SNE on test embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()

    return True, f"Saved to {save_path}"


def _dataset_to_feature_label_arrays(
    lightning_model: torch.nn.Module,
    dataset,
    batch_size: int = 128,
):
    """Extract feature/label arrays from a TensorDataset-like object."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats, labels = [], []

    was_training = lightning_model.training
    lightning_model.eval()

    try:
        device = next(lightning_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = _extract_features(lightning_model, x)
            feats.append(f.detach().cpu())
            labels.append(y.detach().cpu())

    if was_training:
        lightning_model.train()

    if not feats:
        return None, None

    X = torch.cat(feats, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy().astype(int)
    return X, y


def plot_tsne_source_target(
    lightning_model: torch.nn.Module,
    source_dataset,
    target_dataset,
    save_path,
    class_names=None,
    max_samples: int = 3000,
    random_state: int = 42,
    class_palette=None,
):
    """Create and save a source+target joint t-SNE figure.

    Colors encode class labels; marker shapes encode domains.
    """
    Xs, ys = _dataset_to_feature_label_arrays(lightning_model, source_dataset)
    Xt, yt = _dataset_to_feature_label_arrays(lightning_model, target_dataset)

    if Xs is None or Xt is None:
        return False, "No features collected from source/target datasets"

    if max_samples is not None and Xs.shape[0] + Xt.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        src_cap = min(Xs.shape[0], int(max_samples * 0.7))
        tgt_cap = min(Xt.shape[0], max_samples - src_cap)
        if tgt_cap <= 0:
            tgt_cap = min(Xt.shape[0], max(1, max_samples // 3))
            src_cap = min(Xs.shape[0], max_samples - tgt_cap)
        src_idx = rng.choice(Xs.shape[0], size=src_cap, replace=False)
        tgt_idx = rng.choice(Xt.shape[0], size=tgt_cap, replace=False)
        Xs, ys = Xs[src_idx], ys[src_idx]
        Xt, yt = Xt[tgt_idx], yt[tgt_idx]

    X = np.concatenate([Xs, Xt], axis=0)
    y = np.concatenate([ys, yt], axis=0)
    domain = np.concatenate([
        np.zeros(Xs.shape[0], dtype=int),
        np.ones(Xt.shape[0], dtype=int),
    ])

    n_samples = X.shape[0]
    if n_samples < 3:
        return False, f"Not enough samples for t-SNE: {n_samples}"

    perplexity = min(30, max(5, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
    if perplexity < 2:
        return False, f"Invalid t-SNE perplexity {perplexity} for {n_samples} samples"

    if class_palette is None and class_names is not None and len(class_names) == 4:
        class_palette = ["#E53935", "#43A047", "#FDD835", "#1E88E5"]

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    Z = reducer.fit_transform(X)

    classes = np.unique(y)
    cmap = plt.get_cmap("tab10", max(len(classes), 1))

    plt.figure(figsize=(8.2, 6.2))
    for i, cls in enumerate(classes):
        if class_palette is not None and int(cls) < len(class_palette):
            c = class_palette[int(cls)]
        else:
            c = cmap(i)

        src_mask = (y == cls) & (domain == 0)
        tgt_mask = (y == cls) & (domain == 1)

        plt.scatter(
            Z[src_mask, 0],
            Z[src_mask, 1],
            s=20,
            alpha=0.9,
            color=c,
            marker="o",
            edgecolors="black",
            linewidths=0.25,
        )
        plt.scatter(
            Z[tgt_mask, 0],
            Z[tgt_mask, 1],
            s=26,
            alpha=0.95,
            color=c,
            marker="^",
            edgecolors="black",
            linewidths=0.25,
        )

    class_handles = []
    for i, cls in enumerate(classes):
        label = str(int(cls))
        if class_names is not None and int(cls) < len(class_names):
            label = class_names[int(cls)]
        if class_palette is not None and int(cls) < len(class_palette):
            c = class_palette[int(cls)]
        else:
            c = cmap(i)
        class_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                                    markeredgecolor="black", markeredgewidth=0.25,
                                    markersize=7, label=label))

    domain_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markersize=7,
               linestyle="", label="Source"),
        Line2D([0], [0], marker="^", color="black", markerfacecolor="white", markersize=7,
               linestyle="", label="Target"),
    ]

    leg1 = plt.legend(handles=class_handles, loc="upper right", fontsize=8, title="Class")
    plt.gca().add_artist(leg1)
    plt.legend(handles=domain_handles, loc="lower right", fontsize=8, title="Domain")

    plt.title("t-SNE on source+target embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()

    return True, f"Saved to {save_path}"
