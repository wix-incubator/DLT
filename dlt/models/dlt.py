import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from models.utils import PositionalEncoding, TimestepEmbedder


class DLT(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, categories_num, latent_dim=256, num_layers=4, num_heads=4, dropout_r=0., activation="gelu",
                 cond_emb_size=224, cat_emb_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.categories_num = categories_num
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, self.dropout_r)
        # learnable embedding for each category.
        self.cat_emb = nn.Parameter(torch.randn(self.categories_num, cat_emb_size))
        # condition embedding
        self.cond_mask_box_emb = nn.Parameter(torch.randn(2, cond_emb_size))
        self.cond_mask_cat_emb = nn.Parameter(torch.randn(2, cat_emb_size))

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=self.latent_dim * 2,
                                                          dropout=dropout_r,
                                                          activation=activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.seq_pos_enc)

        self.output_process = nn.Sequential(
            nn.Linear(self.latent_dim, 4))

        self.output_cls = nn.Sequential(
            nn.Linear(self.latent_dim, categories_num))

        self.size_emb = nn.Sequential(
            nn.Linear(2, cond_emb_size),
        )

        self.loc_emb = nn.Sequential(
            nn.Linear(2, cond_emb_size),
        )

    def forward(self, sample, noisy_sample, timesteps):
        # put noize to element categories, those we want to predict
        cat_input = noisy_sample['cat'] * sample['mask_cat'] + (1 - sample['mask_cat']) * sample['cat']
        cat_input_flat = rearrange(cat_input, 'b c -> (b c)')
        # pit noize to element boxes, those we want to predict
        sample_tensor = sample['mask_box'] * noisy_sample['box'] + (1 - sample['mask_box']) * sample['box_cond']

        xy = sample_tensor[:, :, :2]
        wh = sample_tensor[:, :, 2:]

        elem_cat_emb = self.cat_emb[cat_input_flat, :]
        elem_cat_emb = rearrange(elem_cat_emb, '(b c) d -> b c d', b=noisy_sample['box'].shape[0])

        mask_wh = sample['mask_box'][:, :, 2]
        mask_xy = sample['mask_box'][:, :, 0]

        def mask_to_emb(mask, cond_mask_emb):
            mask_flat = rearrange(mask, 'b c -> (b c)').type(torch.LongTensor)
            mask_all_emb = cond_mask_emb[mask_flat, :]
            mask_all_emb = rearrange(mask_all_emb, '(b c) d -> b c d', b=mask.shape[0])
            return mask_all_emb

        emb_mask_wh = mask_to_emb(mask_wh, self.cond_mask_box_emb)
        emb_mask_xy = mask_to_emb(mask_xy, self.cond_mask_box_emb)
        emb_mask_cl = mask_to_emb(sample['mask_cat'], self.cond_mask_cat_emb)
        t_emb = self.embed_timestep(timesteps)

        size_emb = self.size_emb(wh) + emb_mask_wh
        loc_emb = self.loc_emb(xy) + emb_mask_xy
        elem_cat_emb = elem_cat_emb + emb_mask_cl

        tokens_emb = torch.cat([size_emb, loc_emb, elem_cat_emb], dim=-1)
        tokens_emb = rearrange(tokens_emb, 'b c d -> c b d')
        # adding the timestep embed
        xseq = torch.cat((t_emb, tokens_emb), dim=0)
        xseq = self.seq_pos_enc(xseq)

        output = self.seqTransEncoder(xseq)[1:]
        output = rearrange(output, 'c b d -> b c d')
        output_box = self.output_process(output)
        output_cls = self.output_cls(output)
        return output_box, output_cls
