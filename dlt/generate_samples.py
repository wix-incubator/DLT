
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data_loaders.magazine import MagazineLayout
from logger_set import LOG
from absl import flags, app
from diffusion import JointDiffusionScheduler
from ml_collections import config_flags
from models.dlt import DLT
from utils import set_seed, draw_layout_opacity
from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test2', help="Work unit directory.")
flags.DEFINE_string("epoch", default='399', help="Epoch to load from checkpoint.")
flags.DEFINE_string("cond_type", default='all', help="Condition type to sample from.")
flags.DEFINE_bool("save", default=False, help="Save samples.")
flags.mark_flags_as_required(["config"])


def sample_from_model(sample, model, device, categories_num, diffusion):
    shape = sample['box_cond'].shape
    model.eval()

    # generate initial noise
    noisy_batch = {
        'box': torch.randn(*shape, dtype=torch.float32, device=device),
        'cat': (categories_num - 1) * torch.ones((shape[0], shape[1]), dtype=torch.long, device=device),
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(diffusion.num_cont_steps)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            # denoise for step t.
            bbox_pred, cat_pred = model(sample, noisy_batch, timesteps=t)
            desc_pred = {
                'cat': cat_pred,
            }
            # sample
            bbox_pred, cat_pred = diffusion.step_jointly(bbox_pred, desc_pred,
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['box'])
            # update noise with x_t + update denoised categories.
            noisy_batch['box'] = bbox_pred.prev_sample
            noisy_batch['cat'] = cat_pred['cat']
    return bbox_pred.pred_original_sample, cat_pred['cat']


def main(*args, **kwargs):
    config = init_job()
    config.optimizer.batch_size = 64
    LOG.info("Loading data.")
    if config.dataset == 'publaynet':
        val_data = PublaynetLayout(config.val_json, 9, config.cond_type)
    elif config.dataset == 'rico':
        val_data = RicoLayout(config.dataset_path, 'test', 9, config.cond_type)
    elif config.dataset == 'magazine':
        val_data = MagazineLayout(config.val_json, 16, config.cond_type)
    else:
        raise NotImplementedError
    assert config.categories_num == val_data.categories_num

    LOG.info("Creating model and diffusion process...")
    model = DLT.from_pretrained(config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}', strict=True)
    model.to(config.device)
    noise_scheduler = JointDiffusionScheduler(alpha=0.0,
                                              seq_max_length=config.max_num_comp,
                                              device=config.device,
                                              discrete_features_names=[('cat', config.categories_num), ],
                                              num_discrete_steps=[config.num_discrete_steps, ],
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type='sample',
                                              clip_sample=False, )

    val_loader = DataLoader(val_data, batch_size=config.optimizer.batch_size,
                            shuffle=False, num_workers=config.optimizer.num_workers)
    model.eval()

    all_results = {
        'dataset_val': [],
        'predicted_val': []
    }

    i = 0

    for batch in tqdm(val_loader):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            bbox_pred, cat_pred = sample_from_model(batch, model, config.device,
                                                    config.categories_num, noise_scheduler)
        # save samples
        box = batch['mask_box'] * bbox_pred + (1 - batch['mask_box']) * batch['box_cond']
        cat = batch['mask_cat'] * cat_pred + (1 - batch['mask_cat']) * batch['cat']
        box = box.cpu().numpy()
        cat = cat.cpu().numpy()

        all_results['dataset_val'].append(
            np.concatenate([batch['box_cond'].cpu().numpy(),
                            np.expand_dims(batch['cat'].cpu().numpy(), -1)], axis=-1))
        all_results['predicted_val'].append(
            np.concatenate([box, np.expand_dims(cat, -1)], axis=-1))
        if config.save:
            for b in range(box.shape[0]):
                tmp_box = box[b]
                tmp_cat = cat[b]
                tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
                tmp_cat = tmp_cat[~(tmp_cat == 0)]
                tmp_box = (tmp_box / 2 + 1) / 2
                canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
                Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}.jpg')
                tmp_box = batch['box_cond'][b].cpu().numpy()
                tmp_cat = batch['cat'][b].cpu().numpy()
                tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
                tmp_cat = tmp_cat[~(tmp_cat == 0)]
                tmp_box = (tmp_box / 2 + 1) / 2
                canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
                Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}_gt.jpg')
        i += 1
    # pickle results
    with open(config.optimizer.samples_dir / f'results_{config.cond_type}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    config.epoch = FLAGS.epoch
    set_seed(config.seed)
    assert config.dataset in ['publaynet', 'rico', 'magazine']
    assert FLAGS.cond_type in ['whole_box', 'loc', 'all']
    config.cond_type = FLAGS.cond_type
    config.save = FLAGS.save
    return config


if __name__ == '__main__':
    app.run(main, )
