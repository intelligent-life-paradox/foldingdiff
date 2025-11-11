from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertConfig
import pytorch_lightning as pl
from simple_impl.bert_for_diffusion import BertForDiffusion
from foldingdiff.datasets import NoisedAnglesDataset, CathCanonicalAnglesOnlyDataset

max_seq_len = 128
min_seq_len = 40

ds_args = dict(
    pad=max_seq_len,
    min_length=min_seq_len,
    trim_strategy="leftalign",
    zero_center=True,
    toy=None,
    pdbs="cath",
)
train_dataset = CathCanonicalAnglesOnlyDataset(split="train", **ds_args)
val_dataset = CathCanonicalAnglesOnlyDataset(split="validation", **ds_args)

exhaustive_t = False
noised_ds_args = dict(
    dset_key="angles",
    timesteps=1000,
    exhaustive_t=False,
    beta_schedule="linear",
    nonangular_variance=1.0,
    angular_variance=np.pi,
)
train_noised_dataset = NoisedAnglesDataset(dset=train_dataset, **noised_ds_args)
val_noised_dataset = NoisedAnglesDataset(dset=val_dataset, **noised_ds_args)

train_dataloader = DataLoader(
    dataset=train_noised_dataset,
    batch_size=64,
    shuffle=True,  # Shuffle only train loader
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    dataset=val_noised_dataset,
    batch_size=64,
    shuffle=False,  # Shuffle only train loader
    num_workers=4,
    pin_memory=True,
)

cfg = BertConfig(
    max_position_embeddings=max_seq_len,
    num_attention_heads=12,
    hidden_size=384,
    intermediate_size=768,
    num_hidden_layers=12,
    position_embedding_type="relative_key",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    use_cache=False,
)

model = BertForDiffusion(
    config=cfg,
    time_encoding="gaussian_fourier",
    decoder="mlp",
    ft_names=train_dataset.feature_names["angles"],
    lr=5e-05,
    loss="smooth_l1",
    l2=0.0,
    l1=0.0,
    epochs=10000,
    steps_per_epoch=len(train_dataloader),
)

results_folder = Path("./results")

trainer = pl.Trainer(
    default_root_dir=results_folder,
    gradient_clip_val=1.0,
    min_epochs=10000,
    max_epochs=10000,
    check_val_every_n_epoch=1,
    callbacks=None,
    logger=pl.loggers.CSVLogger(save_dir=results_folder / "logs"),
    log_every_n_steps=min(200, len(train_dataloader)),  # Log >= once per epoch
    accelerator="cuda",
    strategy=None,
    gpus=-1,
    enable_progress_bar=False,
    move_metrics_to_cpu=False,  # Saves memory
)

if __name__ == "__main__":
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
