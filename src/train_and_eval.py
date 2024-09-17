import sys
sys.path.append('../')
from utils import load_config, show_image_gray, KerasProgressBar, getInputArguments, rich_progress_bar, show_images
import pytorch_lightning as pl
from datasets.datamodule_factory import CommonDatamodule
from model import MetricLDN
from callbacks import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
import torch
torch.set_float32_matmul_precision('high')
from torchsummary import summary


# --- Gather arguments
args = getInputArguments(sys.argv)
cfg = load_config(args.cfg)

# --- Create Datamodule
test_pkl = cfg['lunar_trajPkl'] if args.dataset == 'lunar' else None
datamodule = CommonDatamodule(args.dataset, cfg[args.dataset], cfg[f'{args.dataset}_ltype'], test_pkl=test_pkl, nper_class=2, batch_size=args.batch_size, img_sz=320)
    
# Create model
args.datamodule = datamodule
model = MetricLDN(args=args)

# --- Setup experiment name
# model_name = f'{str(model)}'
model_name = 'cosplace_multisim'
print(f'\nModel Name:\n{model_name}\n')

# --- Print out model architecture
# summary(model.feature_extractor.cuda(), (3,128,128))
# print(model.feature_extractor)

# --- Setup training variables and callbacks
callbacks=[
    KerasProgressBar(),
    EarlyStopping(args.ml_loss, patience=args.epochs, verbose=False, check_finite=True), # NaN checker
    # rich_progress_bar(),
    # ----- End of training eval callbacks
    # AttVisGif(),
    # AttVis(),
    # AttMAE(),
    # CamVisGif(),
    # CamVis(),
    # CamMAE(),
    # LogAccuracies(),
    # Eval_RecallAtKCallback([1], num_galleries=[1], gpu=True),
    # Eval_TestSetRecogCallback(datamodule, transform_type='ill',   repetitions=[2], img_sz=320),
    # Eval_TestSetRecogCallback(datamodule, transform_type='trans', repetitions=[2], img_sz=320),
    # Eval_TestSetRecogCallback(datamodule, transform_type='rot',   repetitions=[2], img_sz=320),
    # Eval_TestSetRecogCallback(datamodule, transform_type='all',   repetitions=[2], img_sz=320),
]

# # If hirise/lunar datasets, add blender navigation experiments
if args.dataset == 'hirise' or args.dataset == 'lunar':
    callbacks += [
        # Eval_BlenderRecogCallback(cfg['lro_traj_anno'], 'ill',   img_sz=320),
        # Eval_BlenderRecogCallback(cfg['lro_traj_anno'], 'trans', img_sz=320),
        # Eval_BlenderRecogCallback(cfg['lro_traj_anno'], 'rot',   img_sz=320),
        Eval_BlenderRecogCallback(cfg['lro_traj_anno'], 'all',   img_sz=320),
        # Eval_BlenderLISCallback(  cfg['lro_traj_anno'], 'ill',   img_sz=320),
        # Eval_BlenderLISCallback(  cfg['lro_traj_anno'], 'trans', img_sz=320),
        # Eval_BlenderLISCallback(  cfg['lro_traj_anno'], 'rot',   img_sz=320),
        # Eval_BlenderLISCallback(  cfg['lro_traj_anno'], 'all',   img_sz=320),
    ]

# --- Setup directories and W&B Logger
train_log_dir = cfg['logdir'] + f'/{model_name}'
wandb_log_dir = cfg['logdir'] + '/wandb/'
Path(train_log_dir).mkdir(parents=True, exist_ok=True)
Path(wandb_log_dir).mkdir(parents=True, exist_ok=True)

if not args.nologger:
    wandb_logger = WandbLogger(
        name=f'{model_name}', 
        project=f'cvpr_24_rebuttal', 
        save_dir=wandb_log_dir, 
        log_model=True
    )
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.watch(model, log='all', log_graph=True)

# --- Train
trainer = pl.Trainer(
    # limit_train_batches=10,
    default_root_dir=train_log_dir, 
    max_epochs=args.epochs, 
    accelerator='auto', 
    callbacks=callbacks,
    enable_checkpointing=args.save,
    logger=True if args.nologger else wandb_logger,
    log_every_n_steps=1,
)
trainer.fit(model, datamodule, ckpt_path=args.ckpt_path if args.ckpt_path else None)
