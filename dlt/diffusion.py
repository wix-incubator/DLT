from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from einops import rearrange
from labml_nn.sampling import Sampler
from torch.distributions import Categorical


class TemperatureSampler(Sampler):
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        # Create a categorical distribution with temperature adjusted logits
        dist = Categorical(probs=logits / self.temperature)

        # Sample
        return dist.sample()


class JointDiffusionScheduler(DDPMScheduler):
    def __init__(self, alpha=0.1, beta=0.15, seq_max_length=16, device='cpu',
                 discrete_features_names: List[Tuple[str, int]] = None,
                 num_discrete_steps: List[int] = None,
                 *args, **kwargs):
        """
        :param alpha: probability to change category for discrete diffusion.
        :param beta: probability beta category is the same, 1 - beta is the probability to change [MASK].
        :param seq_max_length: max number of elements in the sequence.
        :param device:
        :param discrete_features_names: list of tuples (feature_name, number of categories)
        :param num_discrete_steps: num steps for discrete diffusion.
        :param args: params for DDPMScheduler
        :param kwargs: params for DDPMScheduler
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_cont_steps = kwargs['num_train_timesteps']
        if discrete_features_names:
            assert len(discrete_features_names) == len(num_discrete_steps), ("for each feature should be number of "
                                                                             "discrete steps")
            self.discrete_features_names = discrete_features_names
            self.num_discrete_steps = num_discrete_steps
            self.beta = beta
            self.alpha = alpha
            self.seq_max_length = seq_max_length
            self.cont2disc = {}
            self.transition_matrices = {}
            # markov transition matrix for each step for efficiency.
            for tmp, f_steps in zip(discrete_features_names, num_discrete_steps):
                f_name, f_cat_num = tmp
                self.cont2disc[f_name] = self.mapping_cont2disc(self.num_cont_steps, f_steps)
                self.transition_matrices[f_name] = self.generate_transition_mat(f_cat_num, f_steps)

        self.sampler = TemperatureSampler(temperature=0.8)

    def add_noise_jointly(self, vec_cont: torch.FloatTensor, vec_cat: dict,
                          timesteps: torch.IntTensor, noise: torch.FloatTensor) -> Tuple[torch.FloatTensor, dict]:
        """
        Forward diffusion process for continuous and discrete features.
        :param vec_cont: continuous feature
        :param vec_cat: dict for all discrete features
        :param timesteps: diffusion timestep
        :param noise: noise for continuous feature.
        :return: tuple of  noised continuous feature and noised discrete features.
        """
        noised_cont = super().add_noise(original_samples=vec_cont, timesteps=timesteps, noise=noise)
        cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:
            t_to_discrete_stage = [self.cont2disc[f_name][t.item()] for t in timesteps]
            prob_mat = [self.transition_matrices[f_name][u][vec_cat[f_name][i]] for i, u in enumerate(t_to_discrete_stage)]
            prob_mat = torch.cat(prob_mat)
            cat_noise = torch.multinomial(prob_mat, 1, replacement=True)
            cat_noise = rearrange(cat_noise, '(d b) 1 -> d b', d=noised_cont.shape[0])
            cat_res[f_name] = cat_noise
        return noised_cont, cat_res

    def step_jointly(self, cont_output: torch.FloatTensor, cat_output: dict, timestep, sample: torch.FloatTensor,
                     generator=None,
                     return_dict: bool = True, ):
        """Reverse diffusion process for continuous and discrete features."""
        bbox = super().step(cont_output, timestep.detach().item(), sample, generator, return_dict)
        step_cat_res = {}
        for f_name, f_cat_num in self.discrete_features_names:
            t_to_discrete_stage = [self.cont2disc[f_name][t.item()] for t in timestep]
            cls, _ = self.denoise_cat(cat_output[f_name], t_to_discrete_stage,
                                      f_cat_num, self.transition_matrices[f_name])
            step_cat_res[f_name] = cls
        return bbox, step_cat_res

    def generate_transition_mat(self, categories_num, num_discrete_steps):
        """Markov transition matrix for discrete diffusion."""
        transition_mat = np.eye(categories_num) * (1 - self.alpha - self.beta) + self.alpha / categories_num
        transition_mat[:, -1] += self.beta
        transition_mat[-1, :] = 0
        transition_mat[-1, -1] = 1
        transition_mat_list = []
        curr_mat = transition_mat.copy()
        for i in range(num_discrete_steps):
            transition_mat_list.append(torch.tensor(curr_mat).to(torch.float32).to(self.device))
            curr_mat = curr_mat @ transition_mat
        return transition_mat_list

    def denoise_cat(self, pred, t, cat_num, transition_mat_list):
        pred_prob = F.softmax(pred, dim=2)
        prob, cls = torch.max(pred_prob, dim=2)

        if t[0] > 1:
            m = torch.matmul(pred_prob.reshape((-1, cat_num)),
                             transition_mat_list[t[0]].to(self.device).float())
            m = m.reshape(pred_prob.shape)
            m[:, :, 0] = 0
            res = self.sampler(m)
        else:
            res = (cat_num - 1) * torch.ones_like(cls).to(torch.long)
            top = torch.topk(prob, prob.shape[1], dim=1)
            for ttt in range(prob.shape[0]):
                res[ttt, top[1][ttt]] = cls[ttt, top[1][ttt]]
        return res, 0

    @staticmethod
    def mapping_cont2disc(num_cont_steps, num_discrete_steps):
        block_size = num_cont_steps // num_discrete_steps
        cont2disc = {}
        for i in range(num_cont_steps):
            if i >= (num_discrete_steps - 1) * block_size:
                if num_cont_steps % num_discrete_steps != 0 and i >= num_discrete_steps * block_size:
                    cont2disc[i] = num_discrete_steps - 1
                else:
                    cont2disc[i] = i // block_size
            else:
                cont2disc[i] = i // block_size
        return cont2disc
