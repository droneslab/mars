from pytorch_grad_cam import EigenCAM
import wandb
import sys
sys.path.append('../src/')
from model import MetricLDN
from datasets.datamodule_factory import CommonDatamodule
from utils import load_config, show_images, getInputArguments
from pytorch_lightning.loggers import WandbLogger
from eval import Eval_RecallAtK, Eval_TestSetRecogAccuracy, Eval_BlenderLRORecogAccuracy, Eval_BlenderLROLostInSpace
import torch
torch.set_float32_matmul_precision('high')
from glob import glob

def DetachedRecall(logger, model, test_dataloader, ks, num_galleries, gpu=False):        
        print(f"\n----- Recall@Ks")
        print(f"  Ks:            {ks}")
        print(f"  Gallery sizes: {num_galleries}")
        dl = test_dataloader
        if gpu:
            model = model.cuda().eval()
        else:
            model = model.cpu().eval()
            
        for gallery_size in num_galleries:
            Eval_RecallAtK(logger, model, dl, ks, gallery_size, gpu=gpu)
            
def DetachedTestSetRecog(logger, model, datamodule, transform_type, repetitions, thresh=0.9, gpu=True):        
        print(f"\n----- Test Set: Recognition Accuracy ({transform_type})")
        print(f"  thresh:      {thresh}")
        print(f"  use_gpu:     {gpu}")
        print(f"  repetitions: {repetitions}")
        dm = datamodule
        if gpu:
            model = model.cuda().eval()
        else:
            model = model.cpu().eval()
        
        for rep in repetitions:
            print(f'--- rep {rep}')
            dl = dm.make_new_test_ds_dl(transform_type=transform_type, repetitions=rep, sampler=False)
            Eval_TestSetRecogAccuracy(logger,model,dl,transform_type, use_gpu=gpu, reps=rep)
            
def DetachedBlenderRecog(logger, model, anno_file, transType, use_gpu=False):
        print(f"\n----- Blender LRO Trajectory: Recognition Accuracy ({transType})")
        print(f"  anno_file:        {anno_file}")
        if use_gpu:
            model = model.cuda().eval()
        else:
            model = model.cpu().eval()
        Eval_BlenderLRORecogAccuracy(logger, model, anno_file, gpu=use_gpu, transType=transType)                                     

def DetachedBlenderLIS(logger, model, anno_file, transType, use_gpu=False):
        print(f"\n----- Blender LRO Trajectory: Lost In Space Accuracy")
        print(f"  anno_file:        {anno_file}")
        print(f"  transType:        {transType}")
        if use_gpu:
            model = model.cuda().eval()
        else:
            model = model.cpu().eval()
            
        Eval_BlenderLROLostInSpace(logger, model, anno_file, gpu=use_gpu, transType=transType)
            
if __name__ == '__main__':
    
    hirise_failure_ckpts = glob('../data/training_logs/failures/hirise/*.ckpt')
    lunar_failure_ckpts = glob('../data/training_logs/failures/lunar/*.ckpt')
    
    for ckpt_p in hirise_failure_ckpts:
        if 'circle' in ckpt_p:
            continue
        
        run_id = ckpt_p.split('/')[-1].split('_')[0]
        wandb.init(entity='tjchase34', project='cvpr_24', id=run_id, resume='must')
        
        # --- Gather arguments
        args = getInputArguments(sys.argv)
        cfg = load_config(args.cfg)
        
        ml_loss = ckpt_p.split('_')[-1].split('.')[0]
        args.ml_loss = ml_loss

        # --- Create Datamodule
        test_pkl = cfg['lunar_trajPkl'] if args.dataset == 'lunar' else None
        datamodule = CommonDatamodule(args.dataset, cfg[args.dataset], cfg[f'{args.dataset}_ltype'], test_pkl=test_pkl, nper_class=2, batch_size=args.batch_size)

        # Create model
        args.datamodule = datamodule
        model = MetricLDN.load_from_checkpoint(ckpt_p, args=args).eval().cuda()
        
        # --- Setup experiment name
        model_name = f'{str(model)}'
        print(f'\nModel Name:\n{model_name}\n')

        wandb_logger = WandbLogger(
            name=f'{model_name}', 
            project=f'cvpr_24', 
            save_dir='./reeval/', 
            log_model=True
        )
        
        # DetachedRecall(wandb_logger, model, datamodule.test_dataloader(), [1], [1], gpu=False)
        DetachedTestSetRecog(wandb_logger, model, datamodule, transform_type='ill',   repetitions=[2],   gpu=True)
        DetachedTestSetRecog(wandb_logger, model, datamodule, transform_type='trans', repetitions=[2],   gpu=True)
        DetachedTestSetRecog(wandb_logger, model, datamodule, transform_type='rot',   repetitions=[2],   gpu=True)
        DetachedTestSetRecog(wandb_logger, model, datamodule, transform_type='all',   repetitions=[2],   gpu=True)
        
        DetachedBlenderRecog(wandb_logger, model, cfg['lro_traj_anno'], 'ill',   use_gpu=True)
        DetachedBlenderRecog(wandb_logger, model, cfg['lro_traj_anno'], 'trans', use_gpu=True)
        DetachedBlenderRecog(wandb_logger, model, cfg['lro_traj_anno'], 'rot',   use_gpu=True)
        DetachedBlenderRecog(wandb_logger, model, cfg['lro_traj_anno'], 'all',   use_gpu=True)
        
        DetachedBlenderLIS(wandb_logger, model, cfg['lro_traj_anno'], 'ill',   use_gpu=True)
        DetachedBlenderLIS(wandb_logger, model, cfg['lro_traj_anno'], 'trans', use_gpu=True)
        DetachedBlenderLIS(wandb_logger, model, cfg['lro_traj_anno'], 'rot',   use_gpu=True)
        DetachedBlenderLIS(wandb_logger, model, cfg['lro_traj_anno'], 'all',   use_gpu=True)
        
        
        
        quit()
    



# wandb.init()

# api = wandb.Api()

# runs = api.runs(path='tjchase34/cvpr_24', filters={
#   'config.dataset': "resisc45",
#   'config.att_type': 'ca',
#   'config.ml_loss': 'proxyanchor'
# })

# ric_ca_id = None
# mars_id = None
# for run in runs:
#     config = run.config
#     if config['aar']:
#         mars_id = f'model-{run.id}:v0'
#     else:
#         ric_ca_id = f'model-{run.id}:v0'
        
# ric_ca_path = wandb.use_artifact('tjchase34/cvpr_24/' + ric_ca_id).download()
# mars_id = wandb.use_artifact('tjchase34/cvpr_24/' + mars_id).download()
    
# # --- Gather arguments
# cfg = load_config('../cfg/laptop.yaml')
# print(cfg)