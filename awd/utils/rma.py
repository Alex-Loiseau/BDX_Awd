import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class RMA:
    def __init__(self, rma_enc_layers, num_rma_obs, rma_history_size, num_obs, device):
        self.rma_enc_layers = rma_enc_layers
        self.num_rma_obs = num_rma_obs
        self.rma_history_size = rma_history_size
        self.num_obs = num_obs
        self.device = device

        rma_latent_dim = self.rma_enc_layers[-1]
        rma_enc_dims = self.rma_enc_layers[:-1]

        # RMA encoder
        self.rma_encoder = None
        activation = nn.ELU()
        rma_enc_layers = []
        rma_enc_layers.append(nn.Linear(self.num_rma_obs, rma_enc_dims[0]))
        rma_enc_layers.append(activation)
        for l in range(len(rma_enc_dims)):
            if l == len(rma_enc_dims) - 1:
                rma_enc_layers.append(nn.Linear(rma_enc_dims[l], rma_latent_dim))
            else:
                rma_enc_layers.append(nn.Linear(rma_enc_dims[l], rma_enc_dims[l + 1]))
                rma_enc_layers.append(activation)
        self.rma_encoder = nn.Sequential(*rma_enc_layers)
        self.rma_encoder = self.rma_encoder.to(self.device)
        print(f"Priv RMA MLP: {self.rma_encoder}")

        # adaptation module
        self.adaptation_module = None
        adapt_enc_layers = []
        adapt_enc_layers.append(
            nn.Linear(self.num_obs * self.rma_history_size, rma_enc_dims[0])
        )
        adapt_enc_layers.append(activation)
        for l in range(len(rma_enc_dims)):
            if l == len(rma_enc_dims) - 1:
                adapt_enc_layers.append(nn.Linear(rma_enc_dims[l], rma_latent_dim))
            else:
                adapt_enc_layers.append(nn.Linear(rma_enc_dims[l], rma_enc_dims[l + 1]))
                adapt_enc_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adapt_enc_layers)
        self.adaptation_module = self.adaptation_module.to(self.device)
        print(f"Adaptation MLP: {self.adaptation_module}")

        self.adaptation_module_learning_rate = 1.0e-3
        self.adaptation_module_optimizer = optim.Adam(
            self.adaptation_module.parameters(), lr=self.adaptation_module_learning_rate
        )
        self.num_adaptation_module_substeps = 1

    def learn(self, rma_obs, rma_history):
        # regress rma encoder to adaptation module

        # rma_obs shape : (num_envs, num_rma_obs)
        # rma_history shape : (num_envs, rma_history_size, num_obs)
        # input should be (batch, flattened input). batch is num_envs

        # flatten
        rma_history = rma_history.view(-1, self.num_obs * self.rma_history_size)

        for _ in range(self.num_adaptation_module_substeps):
            adaptation_pred = self.adaptation_module(rma_history)
            with torch.no_grad():
                adaptation_target = self.rma_encoder(rma_obs)
            adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

            self.adaptation_module_optimizer.zero_grad()
            adaptation_loss.backward()
            print("loss", adaptation_loss)
            self.adaptation_module_optimizer.step()

    # def step(self, rma_obs, rma_history):
    #     rma_obs = self.rma_encoder(rma_obs)
    #     rma_history = self.adaptation_module(rma_history)
    #     return rma_obs, rma_history
