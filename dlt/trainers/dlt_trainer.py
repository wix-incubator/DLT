import math
import os
import random
import shutil

import numpy as np
import torch
import wandb
from diffusers.pipelines import DDPMPipeline
from PIL import Image
from accelerate import Accelerator
from diffusers import get_scheduler
from einops import repeat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_loaders.data_utils import mask_loc, mask_size, mask_whole_box, mask_random_box_and_cat, mask_all
from diffusion import JointDiffusionScheduler

from logger_set import LOG
from utils import masked_l2, masked_cross_entropy, masked_acc, plot_sample


class TrainLoopDLT:
    def __init__(self, accelerator: Accelerator, model, diffusion: JointDiffusionScheduler, train_data,
                 val_data, opt_conf,
                 log_interval: int,
                 save_interval: int, categories_num: int,
                 device: str = 'cpu',
                 resume_from_checkpoint: str = None, ):
        self.categories_num = categories_num
        self.train_data = train_data
        self.val_data = val_data
        self.accelerator = accelerator
        self.save_interval = save_interval
        self.diffusion = diffusion
        self.opt_conf = opt_conf
        self.log_interval = log_interval
        self.device = device

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf.lr, betas=opt_conf.betas,
                                      weight_decay=opt_conf.weight_decay, eps=opt_conf.epsilon)
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size,
                                  shuffle=True, num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=opt_conf.batch_size,
                                shuffle=False, num_workers=opt_conf.num_workers)
        lr_scheduler = get_scheduler(opt_conf.lr_scheduler,
                                     optimizer,
                                     num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
                                     num_training_steps=(len(train_loader) * opt_conf.num_epochs))
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )
        LOG.info((model.device, self.device))

        self.total_batch_size = opt_conf.batch_size * accelerator.num_processes * opt_conf.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(train_loader) / opt_conf.gradient_accumulation_steps)
        self.max_train_steps = opt_conf.num_epochs * self.num_update_steps_per_epoch

        LOG.info("***** Running training *****")
        LOG.info(f"  Num examples = {len(train_data)}")
        LOG.info(f"  Num Epochs = {opt_conf.num_epochs}")
        LOG.info(f"  Instantaneous batch size per device = {opt_conf.batch_size}")
        LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        LOG.info(f"  Gradient Accumulation steps = {opt_conf.gradient_accumulation_steps}")
        LOG.info(f"  Total optimization steps = {self.max_train_steps}")
        self.global_step = 0
        self.first_epoch = 0
        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint:
            LOG.print(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            last_epoch = int(resume_from_checkpoint.split("-")[1])
            self.global_step = last_epoch * self.num_update_steps_per_epoch
            self.first_epoch = last_epoch
            self.resume_step = 0

    def train(self):
        for epoch in range(self.first_epoch, self.opt_conf.num_epochs):
            self.train_epoch(epoch)
            orig, pred = self.generate_images()
            wandb.log({
                "pred": [wandb.Image(pil, caption=f'pred_{self.global_step}_{i:02d}.jpg')
                         for i, pil in pred],
                "orig": [wandb.Image(pil, caption=f'orig_{self.global_step}.jpg')
                         for i, pil in orig]}, step=self.global_step)

    def sample2dev(self, sample):
        for k, v in sample.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    sample[k][k1] = v1.to(self.device)
            else:
                sample[k] = v.to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        device = self.model.device
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = {}
        for step, batch in enumerate(self.train_dataloader):
            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            self.sample2dev(batch)

            # Sample noise that we'll add to the boxes
            noise = torch.randn(batch['box'].shape).to(device)
            bsz = batch['box'].shape[0]
            # Sample a random timestep for each layout
            t = torch.randint(
                0, self.diffusion.num_cont_steps, (bsz,), device=device
            ).long()

            cont_vec, noisy_batch = self.diffusion.add_noise_jointly(batch['box'], batch, t, noise)
            # rewrite box with noised version, original box is still in batch['box_cond']
            noisy_batch['box'] = cont_vec

            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                boxes_predict, cls_predict = self.model(batch, noisy_batch, t)
                loss_mse = masked_l2(batch['box_cond'], boxes_predict, batch['mask_box'])
                loss_cls = masked_cross_entropy(cls_predict, batch['cat'], batch['mask_cat'])

                loss = (self.opt_conf.lmb * loss_mse + loss_cls).mean()

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            losses.setdefault("mse", []).append(loss_mse.mean().detach().item())
            losses.setdefault("cls", []).append(loss_cls.mean().detach().item())
            acc_cat = masked_acc(batch['cat'].detach(),
                                 cls_predict, batch['mask_cat'].detach())
            losses.setdefault("acc_cat", []).append(acc_cat.mean().detach().item())
            losses.setdefault("loss", []).append(loss.detach().item())

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step}
                progress_bar.set_postfix(**logs)

            if self.global_step % self.log_interval == 0:
                wandb.log({k: np.mean(v) for k, v in losses.items()}, step=self.global_step)
                wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)

        progress_bar.close()
        self.accelerator.wait_for_everyone()

        save_path = self.opt_conf.ckpt_dir / f"checkpoint-{epoch}/"
        # delete folder if we have already 5 checkpoints
        if self.opt_conf.ckpt_dir.exists():
            ckpts = list(self.opt_conf.ckpt_dir.glob("checkpoint-*"))
            # sort by epoch
            ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
            if len(ckpts) > 30:
                LOG.info(f"Deleting checkpoint {ckpts[0]}")
                shutil.rmtree(ckpts[0])
        self.accelerator.save_state(save_path)
        self.model.save_pretrained(save_path)
        LOG.info(f"Saving checkpoint to {save_path}")

    def generate_images(self):
        ixs = range(len(self.val_data))
        all_res = []
        orig = []
        ct = 0

        for ix in np.random.choice(ixs, 5, replace=False):
            box, cat, ind, name = self.val_data.get_data_by_ix(ix)

            orig.append((ct, Image.fromarray(plot_sample((box / 2 + 1) / 2, cat, ind, self.val_data.idx2color_map))))

            if ct == 0:
                mask, full_mask_cat = mask_loc(box.shape, 1.0)
            elif ct == 1:
                mask, full_mask_cat = mask_size(box.shape, 1.0)
            elif ct == 2:
                mask, full_mask_cat = mask_whole_box(box.shape, 1.0)
            elif ct == 3:
                mask, full_mask_cat = mask_random_box_and_cat(box.shape, np.random.uniform(0.5, 1.0, size=1)[0],
                                                              np.random.uniform(0.5, 1.0, size=1)[0])
            elif ct == 4:
                mask, full_mask_cat = mask_all(box.shape)

            box, cat, mask, mask4cat = self.val_data.pad_instance(box, cat, mask, full_mask_cat, self.val_data.max_num_comp)
            sample = {
                "box": torch.tensor(box.astype(np.float32), device=self.device),
                "cat": torch.tensor(cat.astype(int), device=self.device),
                "mask_box": torch.tensor(mask.astype(int), device=self.device),
                "mask_cat": torch.tensor(mask4cat.astype(int), device=self.device),
                "box_cond": torch.tensor(box.copy().astype(np.float32), device=self.device),
            }
            # collate from sample to batch using data loader
            sample_cond = self.val_dataloader.collate_fn([sample])

            predicted = self.sample_from_model(sample_cond)

            box, cat = predicted
            box = sample['mask_box'] * box + (1 - sample['mask_box']) * sample['box_cond']
            cat = sample['mask_cat'] * cat + (1 - sample['mask_cat']) * sample['cat']

            box = box.cpu().numpy()[0]
            cat = cat.cpu().numpy()[0]

            box = box[~(box == 0.).all(1)]
            cat = cat[~(cat == 0)]

            box = (box / 2 + 1) / 2
            canvas = plot_sample(box, cat, None, self.val_data.idx2color_map, height=512)
            all_res.append((ct, Image.fromarray(canvas)))

            ct += 1
            if ct > 5:
                break
        return orig, all_res

    def sample_from_model(self, sample):
        shape = sample['box_cond'].shape
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        # generate initial noise
        noisy_batch = {
            'box': torch.randn(*shape, dtype=torch.float32, device=self.device),
            'cat': (self.categories_num - 1) * torch.ones((shape[0], shape[1]), dtype=torch.long, device=self.device),
        }

        # sample x_0 = q(x_0|x_t)
        for i in range(self.diffusion.num_cont_steps)[::-1]:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                # denoise for step t.
                bbox_pred, cat_pred = model(sample, noisy_batch, timesteps=t)
                desc_pred = {
                    'cat': cat_pred,
                }
                # sample
                bbox_pred, cat_pred = self.diffusion.step_jointly(bbox_pred, desc_pred, timestep=t,
                                                                  sample=noisy_batch['box'])
                # update noise with x_t + update denoised categories.
                noisy_batch['box'] = bbox_pred.prev_sample
                noisy_batch['cat'] = cat_pred['cat']
        return bbox_pred.pred_original_sample, cat_pred['cat']
