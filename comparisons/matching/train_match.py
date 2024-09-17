import sys
sys.path.append('/home/tj/projects/research/landmark_recognition/src')
from utils import load_config, KerasProgressBar, getInputArguments
cfg = load_config("../../cfg.yaml")
import pytorch_lightning as pl
from datasets import BlenderLunarHIRISEData
from matchnet import MatchNet
from checkpoints import epochCkpt

# --- Create Dataset
dataset = BlenderLunarHIRISEData(cfg['lunar_hirise_images'], shuffle=True, sample=True, batch_size=32)

# --- Create Model
model = MatchNet()

# --- Setup experiment name
# exp_name = f'{args.dataset}_{str(model)}_sampler{str(args.use_sampler)}_batch{str(args.batch_size)}_epochs{str(args.epochs)}'
# exp_name = exp_name.replace('((','-').replace(')','').replace('(','-')
# print(f'\nExperiment Name:\n{exp_name}\n')

# --- Train
trainer = pl.Trainer(
    # limit_train_batches=100,
    # default_root_dir='../' + cfg['logdir'] + f'/{exp_name}',
    default_root_dir='../../data/training_logs/latest/lunar_hirise_ResNeXt_se_matchnet_none_samplerTrue_batch32_epochs30',
    max_epochs=30, 
    accelerator='auto', 
    callbacks=[KerasProgressBar(), epochCkpt()],
    enable_checkpointing=True,
)
trainer.fit(model, dataset)
