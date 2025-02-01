import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils.flatten as flatten


class RMA(nn.Module):
    def __init__(self, rma_enc_layers, num_rma_obs, rma_history_size, num_obs, device):
        super().__init__()

        self.rma_enc_layers = rma_enc_layers
        self.num_rma_obs = num_rma_obs
        self.rma_history_size = rma_history_size
        self.num_obs = num_obs
        self.device = device

        self.steps = 0
        self.save_path = "./adaptation_module.pth"

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
        self.steps += 1
        with torch.set_grad_enabled(True):  # not great, but will do

            # rma_obs shape : (num_envs, num_rma_obs)
            # rma_history shape : (num_envs, rma_history_size, num_obs)
            # input should be (batch, flattened input). batch is num_envs

            rma_history = rma_history.flatten(start_dim=1)
            for _ in range(self.num_adaptation_module_substeps):
                adaptation_pred = self.adaptation_module(rma_history)
                with torch.no_grad():
                    adaptation_target = self.rma_encoder(rma_obs)

                adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)
                # adaptation_loss.requires_grad = True  # why do i have to set this ?

                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                if self.steps % 100 == 0:
                    print("loss", adaptation_loss)
                self.adaptation_module_optimizer.step()

        if self.steps % 500 == 0:
            self.save(self.save_path)
            self.export_onnx(f"adaptation_module_{self.steps}.onnx")

    def save(self, path):
        torch.save(self.adaptation_module.state_dict(), path)

    def load(self, path):
        self.adaptation_module.load_state_dict(torch.load(path))

    def export_onnx(self, path):

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                torch.nn.Module.__init__(self)
                self._model = model

            def forward(self, input_dict):
                x = self._model.adaptation_module(input_dict["rma_history"])
                return x

        inputs = {
            "rma_history": torch.randn(1, self.rma_history_size * self.num_obs).to(
                self.device
            )
        }

        with torch.no_grad():
            adapter = flatten.TracingAdapter(
                ModelWrapper(self), inputs, allow_non_tensor=True
            )
            traced = torch.jit.trace(
                adapter, adapter.flattened_inputs, check_trace=False
            )
            # flattened_output = traced(*adapter.flattened_inputs)

        torch.onnx.export(
            traced,
            *adapter.flattened_inputs,
            path,
            verbose=True,
            input_names=["rma_history"],
            output_names=["latent"],
        )
