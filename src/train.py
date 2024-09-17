import sys
sys.path.append('../')
from utils import load_config, KerasProgressBar
import pytorch_lightning as pl
from datasets.datamodule_factory import CommonDatamodule
from model import MetricLDN
from callbacks import *
import torch
torch.set_float32_matmul_precision('high')


# --- Gather arguments, create datamodule, create model
cfg = load_config('../cfg.yaml')
datamodule = CommonDatamodule(cfg['landmarks_dir'], cfg['exclude_indices'], batch_size=cfg['batch_size'], img_sz=cfg['input_size'])
model = MetricLDN(cfg, datamodule)

# --- Setup training variables and callbacks
callbacks=[
    KerasProgressBar(),
]
if cfg['eval']:
    callbacks += [
        Eval_RecallAtKCallback([1], num_galleries=[1], gpu=True), # Recall@1
        Eval_TestSetRecogCallback(datamodule, transform_type='all',   repetitions=[2], img_sz=cfg['input_size']), # Incremental Recall@1
    ]
if cfg['luna1_eval']:
    callbacks += [
        Eval_BlenderRecogCallback(cfg['luna1_annotation_file'], 'all', img_sz=320), # Moon Navigation
        Eval_BlenderLISCallback(cfg['luna1_annotation_file'], 'all',   img_sz=320), # Moon Lost-in-Space
    ]

# --- Train
trainer = pl.Trainer(
    default_root_dir=cfg['logging_dir'], 
    max_epochs=cfg['epochs'], 
    accelerator='auto', 
    callbacks=callbacks,
    enable_checkpointing=True,
    logger=True,
    log_every_n_steps=1,
)
trainer.fit(model, datamodule, ckpt_path=cfg['pretrained_weights'])
