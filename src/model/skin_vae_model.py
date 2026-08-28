from dataclasses import asdict, dataclass
from omegaconf import OmegaConf
from scipy.spatial import cKDTree # type: ignore
from torch import nn, Tensor
from typing import Dict, List

import math
import numpy as np
import random
import torch
import torch.nn.functional as F

from src.rig_package.info.asset import Asset

from .spec import ModelSpec, ModelInput, VaeInput
from .skin_vae.autoencoders import SkinFSQCVAEModel

from .attention_backend import flash_attn_func

class Perceiver(nn.Module):
    def __init__(self, channels, out_tokens, num_heads=8):
        super().__init__()
        self.q_vec = nn.Parameter(torch.randn(out_tokens // num_heads, num_heads, channels) * 0.02)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        k = self.k_proj(x)  # [B, N, C]
        v = self.v_proj(x)  # [B, N, C]
        q_repeated = self.q_vec.repeat(B, 1, 1, 1)
        
        q = q_repeated.view(B, -1, self.num_heads, self.head_dim).type(torch.bfloat16)
        k = k.view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        hidden_states, _ = flash_attn_func(q, k, v)
        hidden_states = hidden_states.view(B, -1, self.num_heads * self.head_dim) # type: ignore
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class SkinVAEModel(ModelSpec):

    def __init__(self, model_config, transform_config, tokenizer_config=None):
        super().__init__(model_config, transform_config, tokenizer_config)
        
        cfg = self.model_config
        self.cond_tokens        = cfg['sample']['cond_tokens']
        self.compress_tokens    = cfg['sample']['compress_tokens']
        self.sample_tokens      = cfg['sample']['sample_tokens']
        self.only_dense         = cfg['sample'].get('only_dense', False)
        self.model_type         = cfg.get('type', 'fsqc')
        
        if self.model_type == 'fsqc':
            self.model = SkinFSQCVAEModel(**cfg['model'], sample_tokens=self.sample_tokens)
        else:
            raise NotImplementedError()
        if self.sample_tokens != self.compress_tokens:
            self.down_perceiver = Perceiver(self.model.latent_channels, self.compress_tokens)
        if self.sample_tokens != self.compress_tokens:
            self.up_perceiver = Perceiver(self.model.latent_channels, self.sample_tokens)
    
    def compile_model(self):
        self.model.compile_model()
    
    @property
    def vocab_size(self) -> int:
        return self.model.FSQ.codebook_size

    @property
    def latent_channels(self) -> int:
        return self.model.latent_channels

    def encode(self, vae_input: VaeInput, num_tokens: int=4, j: int=0, full: bool=False, encode_repeat: int=4, return_cond: bool=True):
        raise NotImplementedError()

    def decode(self, z: Tensor, sampled_cond: Tensor, cond_tokens: Tensor, full: bool=False, encode_repeat: int=4) -> Tensor:
        assert z.shape[0] == sampled_cond.shape[0] == cond_tokens.shape[0]
        if full:
            l = z.shape[0]
            s = []
            for i in range(0, l, encode_repeat):
                t = min(l,i+encode_repeat)
                if self.sample_tokens != self.compress_tokens:
                    _z = self.up_perceiver(z[i:t])
                else:
                    _z = z[i:t]
                logits = self.model._decode(z=_z, cond=cond_tokens[i:t], sampled_points=sampled_cond[i:t])
                s.append(logits)
            return torch.cat(s, dim=0)
        else:
            if self.sample_tokens != self.compress_tokens:
                z = self.up_perceiver(z)
            logits = self.model._decode(z=z, cond=cond_tokens, sampled_points=sampled_cond)
            return logits

    def get_loss_dict(
        self,
        skin_pred: Tensor,
        skin_gt: Tensor,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError()
    
    def get_input(self, batch: Dict) -> VaeInput:
        vertices: Tensor = batch['vertices'].float() # (B, N, 3)
        normals: Tensor = batch['normals'].float() # (B, N, 3)
        uniform_skin: List[Tensor] = batch['uniform_skin'] # [(N, J)]
        dense_skin: List[Tensor] = batch['dense_skin'] # [(J, skin_samples)]
        dense_vertices: List[Tensor] = batch['dense_vertices'] # [(J, skin_samples, 3)]
        dense_normals: List[Tensor] = batch['dense_normals'] # [(J, skin_samples, 3)]
        dense_indices: List[List[int]] = batch['dense_indices'] # [List[J]]
        
        B = vertices.shape[0]
        uniform_cond = torch.cat([vertices, normals], dim=-1).float()
        dense_cond = []
        for i in range(B):
            dense_cond.append(torch.cat([dense_vertices[i], dense_normals[i]], dim=-1).float())
        
        uniform_skin = [s.float() for s in uniform_skin]
        dense_skin = [s.float() for s in dense_skin]
        return VaeInput(
            dense_cond=dense_cond,
            dense_skin=dense_skin,
            dense_indices=dense_indices,
            uniform_cond=uniform_cond,
            uniform_skin=uniform_skin,
        )
    
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def training_step(self, batch: Dict) -> Dict:
        raise NotImplementedError()
    
    def process_fn(self, batch: List[ModelInput], is_train: bool = True) -> List[Dict]:
        res = []
        for b in batch:
            asset = b.asset
            assert asset is not None
            assert asset.sampled_vertex_groups is not None
            assert 'skin' in asset.sampled_vertex_groups
            assert asset.meta is not None
            assert 'dense_indices' in asset.meta
            assert 'dense_skin' in asset.meta
            assert 'dense_vertices' in asset.meta
            assert 'dense_normals' in asset.meta
            _d = {
                'vertices': asset.sampled_vertices,
                'normals': b.asset.sampled_normals,
                'non': {
                    'uniform_skin': asset.sampled_vertex_groups['skin'],
                    'num_bones': asset.J,
                    'skin_samples': asset.skin_samples,
                    'dense_indices': asset.meta['dense_indices'],
                    'dense_skin': asset.meta['dense_skin'],
                    'dense_vertices': asset.meta['dense_vertices'],
                    'dense_normals': asset.meta['dense_normals'],
                }
            }
            res.append(_d)
        return res

    def forward(self, batch: Dict) -> Dict:
        return self.training_step(batch=batch)

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def predict_step(self, batch: Dict) -> Dict:
        vertices: Tensor = batch['vertices'].float() # (B, N, 3)
        num_bones: List[int] = batch['num_bones']
        
        B = vertices.shape[0]
        N = vertices.shape[1]
        
        vae_input = self.get_input(batch=batch)
        num_tokens = 4
        z, cond_tokens, indices, _ = self.encode(vae_input=vae_input, num_tokens=num_tokens, full=True, encode_repeat=8)
        assert cond_tokens is not None
        
        z = self.model.FSQ.indices_to_codes(indices).reshape(z.shape)
        _skin_pred = self.decode(z=z, sampled_cond=vae_input.get_flatten_uniform_cond(), cond_tokens=cond_tokens[vae_input.get_flatten_indices()], full=True, encode_repeat=8)
        _skin_pred = _skin_pred.squeeze(-1)
        
        tot = 0
        results = []
        for i in range(B):
            asset: Asset = batch['model_input'][i].asset.copy()
            skin_pred = torch.zeros((N, num_bones[i]), dtype=vertices.dtype, device=vertices.device)
            for j in range(vae_input.get_len(i=i)):
                skin_pred[:, vae_input.true_j(i=i, j=j)] = _skin_pred[tot]
                tot += 1
            sampled_vertices = vertices[i].detach().float().cpu().numpy()
            tree = cKDTree(sampled_vertices)
            distances, indices = tree.query(asset.vertices)
            sampled_skin = skin_pred.detach().float().cpu().numpy()[indices]
            asset.skin = sampled_skin
            results.append(asset)
        
        return {
            'results': results,
        }