# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization


class StudentTeacher(nn.Module):

    def __init__(
        self,
        obs,
        student_obs,
        num_actions,
        teacher_path,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.obs_s = student_obs
        self.obs_t = obs.shape[1] - student_obs
        self.student = Student(self.obs_s,num_actions)

        # student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(self.obs_s)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # teacher
        self.teacher = torch.jit.load(teacher_path)
        self.teacher.eval()

        # teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(self.obs_t)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        Normal.set_default_validate_args(False)

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.student(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs):
        obs = self.student_obs_normalizer(obs[:,:self.obs_s])
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.student_obs_normalizer(obs[:,:self.obs_s])
        return self.student(obs)

    def evaluate(self, obs):

        obs = self.teacher_obs_normalizer(obs[:,self.obs_s:])
        with torch.no_grad():
            return self.teacher(obs)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        # make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(obs[:,:self.obs_s])
    
class Student(nn.Module):
    def __init__(self, num_prop, 
                 num_actions, 
                 num_scan = 132,
                 scan_encoder_dims=[256, 256, 256],
                 actor_hidden_dims=[256, 256, 256], 
                 activation='elu', 
                 tanh_encoder_output=False) -> None:
        super().__init__()

        self.num_prop = num_prop
        self.num_actions = num_actions
        self.num_scan = num_scan
        activation = get_activation(activation)
        scan_encoder = []
        scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
        scan_encoder.append(activation)
        for l in range(len(scan_encoder_dims) - 1):
            if l == len(scan_encoder_dims) - 2:
                scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                scan_encoder.append(nn.Tanh())
            else:
                scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                scan_encoder.append(activation)
        self.scan_encoder = nn.Sequential(*scan_encoder)
        self.scan_encoder_output_dim = scan_encoder_dims[-1]
        

        actor_input_dim = num_prop - num_scan + self.scan_encoder_output_dim

        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, eval=False, scandots_latent=None):

        obs_prop = obs[:, :self.num_prop - self.num_scan]
        obs_scan = obs[:, (self.num_prop - self.num_scan):]

        if scandots_latent is None:
            scan_latent = self.scan_encoder(obs_scan)   
        else:
            scan_latent = scandots_latent
        
        backbone_input = torch.cat([obs_prop, scan_latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        
        return backbone_output

    def infer_scandots_latent(self, obs):
        if self.num_scan > 0:
            obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
            return self.scan_encoder(obs_scan)
        else:
            return torch.empty(obs.shape[0], 0, device=obs.device)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
