import os, time, yaml, ast, gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics  import MetricsCallback, write_summary
from utils.latency  import measure_latency
from utils.tsne_visualization import plot_tsne_from_test_dataloader, plot_tsne_source_target

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

# Define the path to the configuration directory
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def train_and_test_standard(config):
    """
    标准训练和测试流程（非 CDAN）
    """
    # Create result and checkpoints directories
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = ( 
        Path(__file__).resolve().parent / 
        f"results/{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )   
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves", "tsne"]: (result_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save config to the result directory
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Retrieve model and datamodule classes
    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls(dataset_name)

    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes

    # Parse subject IDs from config
    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg
  
    # Initialize containers for tracking metrics across subjects
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []

    # Loop through each subject ID for training and testing   
    for subject_id in subject_ids:
        print(f"\n>>> Training on subject: {subject_id}")

        # Set seed for reproducibility
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()
   
        # Initialize PyTorch Lightning Trainer
        gpu_id = config.get("gpu_id", 0)
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices = "auto" if gpu_id == -1 else [gpu_id],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy = "auto" if gpu_id != -1 
                else DDPStrategy(find_unused_parameters=True), 
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[metrics_callback]
        )

        # Instantiate datamodule and model
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])

        # Count total number of model parameters
        param_count = sum(p.numel() for p in model.parameters())

        # ---------------- TRAIN ----------------
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60) # minutes

        # ---------------- TEST -----------------
        st_test = time.time()
        test_results = trainer.test(model, datamodule)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        # ---------------- LATENCY --------------
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)

        # ---------------- t-SNE ----------------
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

            ok_st, msg_st = plot_tsne_source_target(
                model,
                datamodule.train_dataset,
                datamodule.test_dataset,
                result_dir / f"tsne/tsne_subject_{subject_id}_source_target.png",
                class_names=datamodule_cls.class_names,
                max_samples=config.get("tsne_max_samples", 2000),
                random_state=config.get("seed", 42),
            )
            if not ok_st:
                print(f"[t-SNE source+target] Subject {subject_id}: {msg_st}")

        # ---------------- METRICS --------------
        test_accs.append(test_results[0]["test_acc"])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])

        # compute & store this subject's confusion matrix
        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        # plot per-subject if requested
        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"Confusion Matrix – Subject {subject_id}",
            )            

        # Plot and save loss and accuracy curves if available
        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                        "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                        "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        # Optionally save the trained model's weights
        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)

        # Free memory between folds
        del model, trainer, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summarize and save final results
    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
        test_accs, test_losses, test_kappas, train_times, test_times, response_times)

    # plot the average if requested
    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names= datamodule_cls.class_names,
            title="Average Confusion Matrix",
        )     


def train_and_test_cdan(config, use_v2=False, use_v2_simple=False, use_cccoral=False, use_scdan=False):
    """
    CDAN 训练和测试流程
    
    使用条件域对抗网络进行跨被试迁移学习
    
    Args:
        config: 配置字典
        use_v2: 是否使用 CDAN v2（改进版本）
        use_v2_simple: 是否使用 CDAN v2 Simple（简化版本）
    """
    from models.atcnet import ATCNetModule
    if use_scdan:
        from models.classification_module_scdan import SCDANClassificationModule as CDANModule
    elif use_cccoral:
        from models.classification_module_cccoral import CDANCCCORALClassificationModule as CDANModule
    elif use_v2_simple:
        from models.classification_module_v2_simple import CDANv2SimpleModule as CDANModule
    elif use_v2:
        from models.classification_module_v2 import CDANv2ClassificationModule as CDANModule
    else:
        from models.classification_module import CDANClassificationModule as CDANModule
    
    # Create result and checkpoints directories
    if use_scdan:
        model_name = config["model"] + "_SCDAN"
    elif use_cccoral:
        model_name = config["model"] + "_CDAN_CCCORAL"
    else:
        model_name = config["model"] + "_CDAN"
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = ( 
        Path(__file__).resolve().parent / 
        f"results/{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )   
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves", "tsne"]: (result_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save config to the result directory
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Retrieve datamodule class (CDAN 专用)
    datamodule_cls = get_datamodule_cls(dataset_name)

    # 设置模型参数
    n_channels = datamodule_cls.channels
    n_classes = datamodule_cls.classes
    d_model = config["model_kwargs"].get("d_model", 32)

    # Parse subject IDs from config
    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg
  
    # Initialize containers for tracking metrics across subjects
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []

    # Loop through each subject ID for training and testing   
    for subject_id in subject_ids:
        print(f"\n>>> CDAN Training - Target subject: {subject_id}")

        # Set seed for reproducibility
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()
   
        # Initialize PyTorch Lightning Trainer
        gpu_id = config.get("gpu_id", 0)
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices = "auto" if gpu_id == -1 else [gpu_id],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy = "auto" if gpu_id != -1 
                else DDPStrategy(find_unused_parameters=True), 
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[metrics_callback]
        )

        # Instantiate datamodule
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)

        # 创建 ATCNet 模型
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

        # 创建 CDAN 训练模块
        if use_scdan:
            cdan_kwargs = config.get("scdan_kwargs", {})
        elif use_v2_simple:
            cdan_kwargs = config.get("cdan_v2_simple", {})
        elif use_v2:
            cdan_kwargs = config.get("cdan_v2", {})
        else:
            cdan_kwargs = config.get("cdan_kwargs", {})
        cccoral_kwargs = config.get("cccoral_kwargs", {})
        
        # 基础参数
        module_params = {
            "model": atcnet_model,
            "n_classes": n_classes,
            "d_model": d_model,
            "discriminator_hidden_dim": cdan_kwargs.get("discriminator_hidden_dim", 256),
            "discriminator_num_layers": cdan_kwargs.get("discriminator_num_layers", 2),
            "discriminator_dropout": cdan_kwargs.get("discriminator_dropout", 0.5),
            "lambda_domain": cdan_kwargs.get("lambda_domain", 1.0),
            "lambda_entropy": cdan_kwargs.get("lambda_entropy", 0.0 if not use_v2 else 0.1),
            "use_entropy_conditioning": cdan_kwargs.get("use_entropy_conditioning", True),
            "use_random_layer": cdan_kwargs.get("use_random_layer", False),
            "random_dim": cdan_kwargs.get("random_dim", 1024),
            "lambda_schedule": cdan_kwargs.get("lambda_schedule", True),
            "lr": config["model_kwargs"].get("lr", 0.0009),
            "lr_discriminator": cdan_kwargs.get("lr_discriminator", None),
            "weight_decay": config["model_kwargs"].get("weight_decay", 0.001),
            "optimizer": config["model_kwargs"].get("optimizer", "adam"),
            "scheduler": config["model_kwargs"].get("scheduler", True),
            "max_epochs": config["max_epochs"],
            "warmup_epochs": config["model_kwargs"].get("warmup_epochs", 20),
            "beta_1": config["model_kwargs"].get("beta_1", 0.5),
            "beta_2": config["model_kwargs"].get("beta_2", 0.999),
        }
        
        # CDAN v2 特有参数
        if use_v2:
            module_params.update({
                "use_spectral_norm": cdan_kwargs.get("use_spectral_norm", True),
                "lambda_contrastive": cdan_kwargs.get("lambda_contrastive", 0.1),
                "use_contrastive": cdan_kwargs.get("use_contrastive", True),
                "lambda_warmup_ratio": cdan_kwargs.get("lambda_warmup_ratio", 0.1),
                "contrastive_temperature": cdan_kwargs.get("contrastive_temperature", 0.07),
            })

        if use_scdan:
            module_params.update({
                "lambda_common": cdan_kwargs.get("lambda_common", 0.1),
                "pseudo_threshold": cdan_kwargs.get("pseudo_threshold", 0.8),
                "min_samples_per_class": cdan_kwargs.get("min_samples_per_class", 2),
            })

        if use_cccoral:
            module_params.update({
                "lambda_cccoral": cccoral_kwargs.get("lambda_cccoral", 0.01),
                "cccoral_alpha": cccoral_kwargs.get("cccoral_alpha", 0.1),
                "pseudo_threshold": cccoral_kwargs.get("pseudo_threshold", 0.9),
                "min_samples_per_class": cccoral_kwargs.get("min_samples_per_class", 4),
                "cccoral_warmup_epochs": cccoral_kwargs.get("cccoral_warmup_epochs", 5),
            })
        
        model = CDANModule(**module_params)

        # Count total number of model parameters
        param_count = sum(p.numel() for p in model.parameters())

        # ---------------- TRAIN ----------------
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60) # minutes

        # ---------------- TEST -----------------
        st_test = time.time()
        test_results = trainer.test(model, datamodule)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        # ---------------- LATENCY --------------
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)

        # ---------------- t-SNE ----------------
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

        # ---------------- METRICS --------------
        test_accs.append(test_results[0]["test_acc"])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])

        # compute & store this subject's confusion matrix
        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        # plot per-subject if requested
        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"CDAN Confusion Matrix – Subject {subject_id}",
            )            

        # Plot and save loss and accuracy curves if available
        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                        "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                        "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        # Optionally save the trained model's weights
        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)

        # Free memory between folds
        del model, trainer, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summarize and save final results
    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
        test_accs, test_losses, test_kappas, train_times, test_times, response_times)
    
    # plot the average if requested
    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names= datamodule_cls.class_names,
            title="CDAN Average Confusion Matrix",
        )     


def train_and_test_coral(config):
    """
    Deep CORAL 训练和测试流程（baseline）。

    Loss = CE(source) + lambda_coral * CORAL(feat_src, feat_tgt)
    无 GRL / 判别器，纯统计对齐。
    """
    from models.atcnet import ATCNetModule
    from models.classification_module_coral import DeepCORALClassificationModule as CORALModule

    model_name = config["model"] + "_DeepCORAL"
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
        print(f"\n>>> Deep CORAL Training - Target subject: {subject_id}")

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
            enable_progress_bar=False,
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

        coral_kwargs = config.get("coral_kwargs", {})
        model = CORALModule(
            model=atcnet_model,
            n_classes=n_classes,
            d_model=d_model,
            lambda_coral=coral_kwargs.get("lambda_coral", 1.0),
            lr=config["model_kwargs"].get("lr", 0.0009),
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
        test_results = trainer.test(model, datamodule)
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
                title=f"Deep CORAL Confusion Matrix – Subject {subject_id}",
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

        # Free memory between folds
        del model, trainer, datamodule
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
            title="Deep CORAL Average Confusion Matrix",
        )


def train_and_test_dann(config):
    """
    DANN 训练和测试流程（baseline）

    使用特征级域对抗（不使用 CDAN 的条件外积映射）。
    """
    from models.atcnet import ATCNetModule
    from models.classification_module_dann import DANNClassificationModule as DANNModule

    model_name = config["model"] + "_DANN"
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
        print(f"\n>>> DANN Training - Target subject: {subject_id}")

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
            enable_progress_bar=False,
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

        dann_kwargs = config.get("dann_kwargs", {})
        module_params = {
            "model": atcnet_model,
            "n_classes": n_classes,
            "d_model": d_model,
            "discriminator_hidden_dim": dann_kwargs.get("discriminator_hidden_dim", 256),
            "discriminator_num_layers": dann_kwargs.get("discriminator_num_layers", 2),
            "discriminator_dropout": dann_kwargs.get("discriminator_dropout", 0.5),
            "lambda_domain": dann_kwargs.get("lambda_domain", 1.0),
            "lambda_entropy": dann_kwargs.get("lambda_entropy", 0.0),
            "lambda_schedule": dann_kwargs.get("lambda_schedule", True),
            "lr": config["model_kwargs"].get("lr", 0.0009),
            "lr_discriminator": dann_kwargs.get("lr_discriminator", None),
            "weight_decay": config["model_kwargs"].get("weight_decay", 0.001),
            "optimizer": config["model_kwargs"].get("optimizer", "adam"),
            "scheduler": config["model_kwargs"].get("scheduler", True),
            "max_epochs": config["max_epochs"],
            "warmup_epochs": config["model_kwargs"].get("warmup_epochs", 20),
            "beta_1": config["model_kwargs"].get("beta_1", 0.5),
            "beta_2": config["model_kwargs"].get("beta_2", 0.999),
        }

        model = DANNModule(**module_params)
        param_count = sum(p.numel() for p in model.parameters())

        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        test_results = trainer.test(model, datamodule)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
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
                title=f"DANN Confusion Matrix – Subject {subject_id}",
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

        # Free memory between folds
        del model, trainer, datamodule
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
            title="DANN Average Confusion Matrix",
        )


# Command-line argument parsing
def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="ATCNet",
        help = "Name of the model to use. Options: ATCNet"
    )        
    parser.add_argument("--dataset", type=str, default="bcic2a", 
        help="Name of the dataset to use. Options: bcic2a, weibo2014"
    )
    parser.add_argument("--loso", action="store_true", default=False, 
        help="Enable subject-independent (LOSO) mode"
    )
    parser.add_argument("--cdan", action="store_true", default=False, 
        help="Enable CDAN (Conditional Domain Adversarial Network) training"
    )
    parser.add_argument("--cdanv2", action="store_true", default=False, 
        help="Enable CDAN v2 (Improved CDAN with attention, contrastive loss, etc.)"
    )
    parser.add_argument("--cdanv2_simple", action="store_true", default=False, 
        help="Enable CDAN v2 Simple (Simplified version with attention only)"
    )
    parser.add_argument("--cccoral", action="store_true", default=False,
        help="Enable baseline CDAN + class-conditional CORAL (CCCORAL)"
    )
    parser.add_argument("--dann", action="store_true", default=False,
        help="Enable DANN baseline training"
    )
    parser.add_argument("--scdan", action="store_true", default=False,
        help="Enable SCDAN training"
    )
    parser.add_argument("--coral", action="store_true", default=False,
        help="Enable Deep CORAL baseline training (CE + CORAL alignment, no adversarial)"
    )
    parser.add_argument("--lambda_coral", type=float, default=None,
        help="CORAL loss weight (default: 1.0)"
    )
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
    
    # CDAN 相关参数
    parser.add_argument("--lambda_domain", type=float, default=None,
                        help="Domain adversarial loss weight for CDAN")
    parser.add_argument("--lambda_entropy", type=float, default=None,
                        help="Entropy minimization loss weight for CDAN")
    parser.add_argument("--no_lambda_schedule", action="store_true",
                        help="Disable lambda scheduling for CDAN")

    # CCCORAL 相关参数（仅在 --cccoral 时生效）
    parser.add_argument("--lambda_cccoral", type=float, default=None,
                        help="Class-conditional CORAL loss weight")
    parser.add_argument("--cccoral_alpha", type=float, default=None,
                        help="Mean alignment weight inside CCCORAL")
    parser.add_argument("--pseudo_threshold", type=float, default=None,
                        help="Pseudo-label confidence threshold for CCCORAL")
    parser.add_argument("--min_samples_per_class", type=int, default=None,
                        help="Minimum samples per class for CCCORAL alignment")
    parser.add_argument("--cccoral_warmup_epochs", type=int, default=None,
                        help="Warmup epochs before enabling CCCORAL loss")
    
    return parser.parse_args()


# ----------------------------------------------
# Main function to run the training and testing pipeline
# ----------------------------------------------
def run():
    args = parse_arguments()

    def _parse_subject_ids(raw: str):
        s = raw.strip()
        if s.lower() == "all":
            return "all"
        # Accept Python-list style, e.g. "[1,2,3]".
        if s.startswith("[") and s.endswith("]"):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
            raise ValueError("--subject_ids list format is invalid")
        # Accept comma-separated format, e.g. "1,2,3".
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        # Single integer subject id.
        return int(s)
     
    # load config
    config_path = os.path.join(CONFIG_DIR, f"{args.model.lower()}.yaml") 
    with open(config_path, encoding='utf-8') as f:    
        config = yaml.safe_load(f)

    # 确定是否使用 CDAN 和版本
    use_cccoral = args.cccoral
    use_dann = args.dann
    use_coral = args.coral
    use_scdan = args.scdan
    use_cdan = args.cdan or args.cdanv2 or args.cdanv2_simple or use_cccoral or use_scdan
    use_cdanv2 = args.cdanv2
    use_cdanv2_simple = args.cdanv2_simple

    if use_cccoral and (use_cdanv2 or use_cdanv2_simple):
        raise ValueError("--cccoral is only supported for baseline CDAN (--cdan), not with --cdanv2/--cdanv2_simple")
    if use_scdan and (use_cccoral or use_cdanv2 or use_cdanv2_simple or args.cdan):
        raise ValueError("--scdan cannot be used together with --cdan/--cdanv2/--cdanv2_simple/--cccoral")
    if use_dann and use_cdan:
        raise ValueError("--dann cannot be used together with CDAN/CCCORAL options")
    if use_coral and (use_cdan or use_dann):
        raise ValueError("--coral cannot be used together with CDAN/DANN options")
    
    # 如果使用 CDAN v2 Simple，加载对应配置
    if use_coral:
        config_path_coral = os.path.join(CONFIG_DIR, f"{args.model.lower()}_cccoral.yaml")
        if os.path.exists(config_path_coral):
            with open(config_path_coral, encoding='utf-8') as f:
                config = yaml.safe_load(f)
    elif use_dann:
        config_path_dann = os.path.join(CONFIG_DIR, f"{args.model.lower()}_dann.yaml")
        if os.path.exists(config_path_dann):
            with open(config_path_dann, encoding='utf-8') as f:
                config = yaml.safe_load(f)
    elif use_scdan:
        config_path_scdan = os.path.join(CONFIG_DIR, f"{args.model.lower()}_scdan.yaml")
        if os.path.exists(config_path_scdan):
            with open(config_path_scdan, encoding='utf-8') as f:
                config = yaml.safe_load(f)
    elif use_cccoral:
        config_path_cccoral = os.path.join(CONFIG_DIR, f"{args.model.lower()}_cccoral.yaml")
        if os.path.exists(config_path_cccoral):
            with open(config_path_cccoral, encoding='utf-8') as f:
                config = yaml.safe_load(f)
    elif use_cdanv2_simple:
        config_path_v2_simple = os.path.join(CONFIG_DIR, f"{args.model.lower()}_cdanv2_simple.yaml")
        if os.path.exists(config_path_v2_simple):
            with open(config_path_v2_simple, encoding='utf-8') as f:
                config = yaml.safe_load(f)
    # 如果使用 CDAN v2，加载对应配置
    elif use_cdanv2:
        config_path_v2 = os.path.join(CONFIG_DIR, f"{args.model.lower()}_cdanv2.yaml")
        if os.path.exists(config_path_v2):
            with open(config_path_v2, encoding='utf-8') as f:
                config = yaml.safe_load(f)

    # Adjust training parameters based on LOSO and CDAN settings
    if args.loso:
        if use_cdan:
            config["dataset_name"] = args.dataset + "_loso_cdan"
        elif use_dann or use_coral:
            config["dataset_name"] = args.dataset + "_loso_cdan"
        else:
            config["dataset_name"] = args.dataset + "_loso"
        config["max_epochs"] = config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]

    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs"]

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]
    
    # Override interaug if specified
    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]

    # EA switch: fallback to config value when present, otherwise default False.
    if args.ea is not None:
        config["preprocessing"]["ea"] = args.ea
    else:
        config["preprocessing"].setdefault("ea", False)

    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    
    # Override seed if specified
    if args.seed is not None:
        config["seed"] = args.seed

    # Override subject IDs if specified
    if args.subject_ids is not None:
        config["subject_ids"] = _parse_subject_ids(args.subject_ids)

    # CDAN 参数配置
    if use_cdan:
        cdan_key = "scdan_kwargs" if use_scdan else "cdan_kwargs"
        cdan_kwargs = config.get(cdan_key, {})
        
        if args.lambda_domain is not None:
            cdan_kwargs["lambda_domain"] = args.lambda_domain
        if args.lambda_entropy is not None:
            cdan_kwargs["lambda_entropy"] = args.lambda_entropy
        if args.no_lambda_schedule:
            cdan_kwargs["lambda_schedule"] = False
            
        config[cdan_key] = cdan_kwargs

    if use_dann:
        dann_kwargs = config.get("dann_kwargs", {})

        if args.lambda_domain is not None:
            dann_kwargs["lambda_domain"] = args.lambda_domain
        if args.lambda_entropy is not None:
            dann_kwargs["lambda_entropy"] = args.lambda_entropy
        if args.no_lambda_schedule:
            dann_kwargs["lambda_schedule"] = False

        config["dann_kwargs"] = dann_kwargs

    if use_cccoral:
        cccoral_kwargs = config.get("cccoral_kwargs", {})

        if args.lambda_cccoral is not None:
            cccoral_kwargs["lambda_cccoral"] = args.lambda_cccoral
        if args.cccoral_alpha is not None:
            cccoral_kwargs["cccoral_alpha"] = args.cccoral_alpha
        if args.pseudo_threshold is not None:
            cccoral_kwargs["pseudo_threshold"] = args.pseudo_threshold
        if args.min_samples_per_class is not None:
            cccoral_kwargs["min_samples_per_class"] = args.min_samples_per_class
        if args.cccoral_warmup_epochs is not None:
            cccoral_kwargs["cccoral_warmup_epochs"] = args.cccoral_warmup_epochs

        config["cccoral_kwargs"] = cccoral_kwargs

    if use_coral:
        coral_kwargs = config.get("coral_kwargs", {})
        if args.lambda_coral is not None:
            coral_kwargs["lambda_coral"] = args.lambda_coral
        config["coral_kwargs"] = coral_kwargs

    # set to True to plot confusion matrices
    config["plot_cm_per_subject"] = True
    config["plot_cm_average"]     = True
    config.setdefault("plot_tsne_per_subject", True)
    config.setdefault("tsne_max_samples", 2000)

    # 根据是否使用 CDAN 选择训练流程
    if use_coral:
        if not args.loso:
            print("Warning: Deep CORAL is designed for LOSO (cross-subject) scenarios. "
                  "Enabling LOSO mode automatically.")
            config["dataset_name"] = args.dataset + "_loso_cdan"
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        train_and_test_coral(config)
    elif use_dann:
        if not args.loso:
            print("Warning: DANN is designed for LOSO (cross-subject) scenarios. "
                  "Enabling LOSO-compatible dataset setting automatically.")
            config["dataset_name"] = args.dataset + "_loso_cdan"
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        train_and_test_dann(config)
    elif use_cdan:
        if not args.loso:
            print("Warning: CDAN is designed for LOSO (cross-subject) scenarios. "
                  "Enabling LOSO mode automatically.")
            config["dataset_name"] = args.dataset + "_loso_cdan"
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        train_and_test_cdan(
            config,
            use_v2=use_cdanv2,
            use_v2_simple=use_cdanv2_simple,
            use_cccoral=use_cccoral,
            use_scdan=use_scdan,
        )
    else:
        train_and_test_standard(config)


if __name__ == "__main__":
    run()
