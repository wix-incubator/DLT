import os
import torch
from logger_set import LOG
from absl import flags, app
from diffusion import JointDiffusionScheduler
from accelerate import Accelerator
import wandb
from ml_collections import config_flags
from models.dlt import DLT
from trainers.dlt_trainer import TrainLoopDLT
from utils import set_seed
from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout
from data_loaders.magazine import MagazineLayout


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(*args, **kwargs):
    config = init_job()
    LOG.info("Loading data.")
    if config.dataset == 'publaynet':
        train_data = PublaynetLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = PublaynetLayout(config.val_json, train_data.max_num_comp)
    elif config.dataset == 'rico':
        train_data = RicoLayout(config.dataset_path, 'train', max_num_comp=config.max_num_comp)
        val_data = RicoLayout(config.dataset_path, 'val', train_data.max_num_comp)
    elif config.dataset == 'magazine':
        train_data = MagazineLayout(config.train_json, max_num_com=config.max_num_comp)
        val_data = MagazineLayout(config.val_json, train_data.max_num_comp)
    else:
        raise NotImplementedError

    assert config.categories_num == train_data.categories_num
    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        mixed_precision=config.optimizer.mixed_precision,
        project_dir=config.log_dir,
    )
    LOG.info(accelerator.state)

    LOG.info("Creating model and diffusion process...")
    model = DLT(categories_num=config.categories_num, latent_dim=config.latent_dim,
                num_layers=config.num_layers, num_heads=config.num_heads, dropout_r=config.dropout_r,
                activation='gelu', cond_emb_size=config.cond_emb_size,
                cat_emb_size=config.cls_emb_size).to(accelerator.device)

    noise_scheduler = JointDiffusionScheduler(alpha=0.0,
                                              seq_max_length=config.max_num_comp,
                                              device=accelerator.device,
                                              discrete_features_names=[('cat', config.categories_num), ],
                                              num_discrete_steps=[config.num_discrete_steps, ],
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type='sample',
                                              clip_sample=False, )

    LOG.info("Starting training...")
    TrainLoopDLT(accelerator=accelerator, model=model, diffusion=noise_scheduler,
                 train_data=train_data, val_data=val_data, opt_conf=config.optimizer,
                 log_interval=config.log_interval, save_interval=config.save_interval,
                 categories_num=train_data.categories_num,
                 device=accelerator.device, resume_from_checkpoint=config.resume_from_checkpoint).train()


def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.optimizer.samples_dir, exist_ok=True)
    os.makedirs(config.optimizer.ckpt_dir, exist_ok=True)
    set_seed(config.seed)
    wandb.init(project='TEST' if FLAGS.workdir == 'test' else 'DLT', name=FLAGS.workdir,
               mode='disabled' if FLAGS.workdir == 'test' else 'online',
               save_code=True, magic=True, config={k: v for k,v in config.items() if k != 'optimizer'})
    wandb.run.log_code(".")
    wandb.config.update(config.optimizer)
    return config


if __name__ == '__main__':
    app.run(main)
