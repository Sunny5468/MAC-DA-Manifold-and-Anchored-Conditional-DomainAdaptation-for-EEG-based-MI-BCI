import os
import time
import yaml
import ast
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from models.atcnet import ATCNetModule
from models.classification_module_daban import MIDABANClassificationModule

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics import MetricsCallback, write_summary
from utils.latency import measure_latency
from utils.tsne_visualization import plot_tsne_from_test_dataloader, plot_tsne_source_target
from utils.get_datamodule_cls import get_datamodule_cls
from utils.seed import seed_everything


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def train_and_test_daban(config):
    """
    MI-DABAN training/testing pipeline.

    Typical architecture:
    - ATCNet feature extractor + classifier
    - global adversarial branch
    - conditional adversarial branch
    - moment alignment branch
    """
    model_name = config["model"] + "_MI_DABAN"
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (
        Path(__file__).resolve().parent /
        f"results/{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves", "tsne"]:
        (result_dir / sub).mkdir(parents=True, exist_ok=True)

    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    datamodule_cls = get_datamodule_cls(dataset_name)
    n_channels = datamodule_cls.channels
    n_classes = datamodule_cls.classes
    d_model = config["model_kwargs"].get("d_model", 32)

    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg

    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []

    for subject_id in subject_ids:
        print(f"\n>>> MI-DABAN Training - Target subject: {subject_id}")

        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()

        gpu_id = config.get("gpu_id", 0)
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices="auto" if gpu_id == -1 else [gpu_id],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto" if gpu_id != -1
                else DDPStrategy(find_unused_parameters=True),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=config.get("progress_bar", False),
            callbacks=[metrics_callback],
        )

        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)

        atcnet_model = ATCNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=config["model_kwargs"].get("F1", 16),
            kernel_length_conv=config["model_kwargs"].get("kernel_length_conv", 64),
            pool_length=config["model_kwargs"].get("pool_length", 8),
            D=config["model_kwargs"].get("D", 2),
            dropout_conv=config["model_kwargs"].get("dropout_conv", 0.3),
            d_model=d_model,
            key_dim=config["model_kwargs"].get("key_dim", 8),
            n_head=config["model_kwargs"].get("n_head", 2),
            dropout_attn=config["model_kwargs"].get("dropout_attn", 0.5),
            tcn_depth=config["model_kwargs"].get("tcn_depth", 2),
            kernel_length_tcn=config["model_kwargs"].get("kernel_length_tcn", 4),
            dropout_tcn=config["model_kwargs"].get("dropout_tcn", 0.3),
            n_windows=config["model_kwargs"].get("n_windows", 5),
        )

        daban_kwargs = config.get("daban_kwargs", {})
        model = MIDABANClassificationModule(
            model=atcnet_model,
            n_classes=n_classes,
            d_model=d_model,
            discriminator_hidden_dim=daban_kwargs.get("discriminator_hidden_dim", 256),
            discriminator_num_layers=daban_kwargs.get("discriminator_num_layers", 2),
            discriminator_dropout=daban_kwargs.get("discriminator_dropout", 0.5),
            lambda_domain=daban_kwargs.get("lambda_domain", 1.0),
            lambda_conditional=daban_kwargs.get("lambda_conditional", 0.5),
            lambda_moment=daban_kwargs.get("lambda_moment", 0.1),
            lambda_entropy=daban_kwargs.get("lambda_entropy", 0.0),
            lambda_schedule=daban_kwargs.get("lambda_schedule", True),
            use_random_layer=daban_kwargs.get("use_random_layer", False),
            random_dim=daban_kwargs.get("random_dim", 1024),
            lr=config["model_kwargs"].get("lr", 0.0009),
            lr_discriminator=daban_kwargs.get("lr_discriminator", None),
            weight_decay=config["model_kwargs"].get("weight_decay", 0.001),
            optimizer=config["model_kwargs"].get("optimizer", "adam"),
            scheduler=config["model_kwargs"].get("scheduler", True),
            max_epochs=config["max_epochs"],
            warmup_epochs=config["model_kwargs"].get("warmup_epochs", 20),
            beta_1=config["model_kwargs"].get("beta_1", 0.5),
            beta_2=config["model_kwargs"].get("beta_2", 0.999),
        )
        param_count = sum(p.numel() for p in model.parameters())

        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        # Reuse current test dataloader to avoid re-triggering datamodule prepare_data
        # inside trainer.test, which can duplicate large CPU buffers.
        test_results = trainer.test(model, dataloaders=datamodule.test_dataloader())
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        lat_ms = measure_latency(model, input_shape, device="cpu")
        response_times.append(lat_ms)

        if config.get("plot_tsne_per_subject", True):
            ok, msg = plot_tsne_from_test_dataloader(
                model,
                datamodule.test_dataloader(),
                result_dir / f"tsne/tsne_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                max_samples=config.get("tsne_max_samples", 2000),
                random_state=config.get("seed", 42),
            )
            if not ok:
                print(f"[t-SNE] Subject {subject_id}: {msg}")

            target_ds = getattr(datamodule, "target_train_dataset", None)
            if target_ds is None:
                target_ds = datamodule.test_dataset

            ok_st, msg_st = plot_tsne_source_target(
                model,
                datamodule.train_dataset,
                target_ds,
                result_dir / f"tsne/tsne_subject_{subject_id}_source_target.png",
                class_names=datamodule_cls.class_names,
                max_samples=config.get("tsne_max_samples", 2000),
                random_state=config.get("seed", 42),
            )
            if not ok_st:
                print(f"[t-SNE source+target] Subject {subject_id}: {msg_st}")

        test_accs.append(test_results[0]["test_acc"])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])

        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"MI-DABAN Confusion Matrix - Subject {subject_id}",
            )

        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                       "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                       "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)

        # Explicitly release per-subject objects to reduce host/GPU memory growth.
        del test_results
        del model
        del atcnet_model
        del datamodule
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
                  test_accs, test_losses, test_kappas, train_times, test_times, response_times)

    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names=datamodule_cls.class_names,
            title="MI-DABAN Average Confusion Matrix",
        )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="ATCNet",
                        help="Name of the model to use. Options: ATCNet")
    parser.add_argument("--dataset", type=str, default="bcic2a",
                        help="Name of the dataset to use. Options: bcic2a")
    parser.add_argument("--loso", action="store_true", default=False,
                        help="Enable subject-independent (LOSO) mode")
    parser.add_argument("--daban", action="store_true", default=False,
                        help="Enable MI-DABAN training")
    parser.add_argument("--progress_bar", action="store_true",
                        help="Enable training progress bar")
    parser.add_argument("--no_progress_bar", action="store_true",
                        help="Disable training progress bar")

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed value (overrides config if specified)")
    parser.add_argument("--subject_ids", type=str, default=None,
                        help="Override subject IDs. Examples: '9', '1,2,3', '[1,2,3]', or 'all'")

    parser.add_argument("--interaug", action="store_true",
                        help="Enable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--no_interaug", action="store_true",
                        help="Disable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--EA", dest="ea", action="store_true",
                        help="Enable Euclidean Alignment preprocessing for BCI2a")
    parser.add_argument("--no_EA", dest="ea", action="store_false",
                        help="Disable Euclidean Alignment preprocessing")
    parser.set_defaults(ea=None)

    parser.add_argument("--lambda_domain", type=float, default=None,
                        help="Global adversarial loss weight")
    parser.add_argument("--lambda_conditional", type=float, default=None,
                        help="Conditional adversarial loss weight")
    parser.add_argument("--lambda_moment", type=float, default=None,
                        help="Moment alignment loss weight")
    parser.add_argument("--lambda_entropy", type=float, default=None,
                        help="Target entropy minimization weight")
    parser.add_argument("--no_lambda_schedule", action="store_true",
                        help="Disable lambda scheduling for adversarial branches")

    parser.add_argument("--use_random_layer", action="store_true",
                        help="Use random layer for conditional branch")
    parser.add_argument("--random_dim", type=int, default=None,
                        help="Projection dimension for random layer")

    return parser.parse_args()


def _parse_subject_ids(raw: str):
    s = raw.strip()
    if s.lower() == "all":
        return "all"
    if s.startswith("[") and s.endswith("]"):
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        raise ValueError("--subject_ids list format is invalid")
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return int(s)


def run():
    args = parse_arguments()

    if args.daban:
        config_path = os.path.join(CONFIG_DIR, f"{args.model.lower()}_daban.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"MI-DABAN config not found: {config_path}. Please add {args.model.lower()}_daban.yaml"
            )
    else:
        config_path = os.path.join(CONFIG_DIR, f"{args.model.lower()}.yaml")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.loso:
        if args.daban:
            config["dataset_name"] = args.dataset + "_loso_cdan"
        else:
            config["dataset_name"] = args.dataset + "_loso"
        config["max_epochs"] = config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"].get("warmup_epochs_loso", 3)
    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs"]

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]

    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]

    if args.ea is not None:
        config["preprocessing"]["ea"] = args.ea
    else:
        config["preprocessing"].setdefault("ea", False)

    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id

    if args.seed is not None:
        config["seed"] = args.seed

    if args.subject_ids is not None:
        config["subject_ids"] = _parse_subject_ids(args.subject_ids)

    if args.daban:
        daban_kwargs = config.get("daban_kwargs", {})

        if args.lambda_domain is not None:
            daban_kwargs["lambda_domain"] = args.lambda_domain
        if args.lambda_conditional is not None:
            daban_kwargs["lambda_conditional"] = args.lambda_conditional
        if args.lambda_moment is not None:
            daban_kwargs["lambda_moment"] = args.lambda_moment
        if args.lambda_entropy is not None:
            daban_kwargs["lambda_entropy"] = args.lambda_entropy
        if args.no_lambda_schedule:
            daban_kwargs["lambda_schedule"] = False
        if args.use_random_layer:
            daban_kwargs["use_random_layer"] = True
        if args.random_dim is not None:
            daban_kwargs["random_dim"] = args.random_dim

        config["daban_kwargs"] = daban_kwargs

    config["plot_cm_per_subject"] = True
    config["plot_cm_average"] = True
    config.setdefault("plot_tsne_per_subject", True)
    config.setdefault("tsne_max_samples", 2000)

    if args.progress_bar:
        config["progress_bar"] = True
    elif args.no_progress_bar:
        config["progress_bar"] = False
    else:
        config.setdefault("progress_bar", False)

    if args.daban:
        if not args.loso:
            print("Warning: MI-DABAN is designed for LOSO (cross-subject) scenarios. "
                  "Enabling LOSO-compatible dataset setting automatically.")
            config["dataset_name"] = args.dataset + "_loso_cdan"
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        train_and_test_daban(config)
    else:
        from train_pipeline import train_and_test_standard
        train_and_test_standard(config)


if __name__ == "__main__":
    run()
