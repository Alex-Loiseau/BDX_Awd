import torch.nn as nn
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils.flatten as flatten
# import pickle
# import numpy as np
import os

# import onnxruntime


# class OnnxInfer:
#     def __init__(self, onnx_model_path, input_name="obs", awd=False):
#         self.onnx_model_path = onnx_model_path
#         self.ort_session = onnxruntime.InferenceSession(
#             self.onnx_model_path, providers=["CPUExecutionProvider"]
#         )
#         self.input_name = input_name
#         self.awd = awd

#     def infer(self, inputs):
#         if self.awd:
#             outputs = self.ort_session.run(None, {self.input_name: [inputs]})
#             return outputs[0][0]
#         else:
#             outputs = self.ort_session.run(
#                 None, {self.input_name: inputs.astype("float32")}
#             )
#             return outputs[0]


class RMA(nn.Module):
    def __init__(
        self, rma_enc_layers, num_rma_obs, rma_history_size, num_obs, device, save_path
    ):
        super().__init__()

        self.rma_enc_layers = rma_enc_layers
        self.num_rma_obs = num_rma_obs
        self.rma_history_size = rma_history_size
        self.num_obs = num_obs
        self.device = device

        self.steps = 0
        self.save_path = save_path

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

        if self.steps % 1000 == 0:
            self.save_adaptation_module()
            self.save_encoder()
            self.export_onnx(f"adaptation_module.onnx")

    def save_adaptation_module(self):
        path = os.path.join(self.save_path, "adaptation_module.pth")
        print("Saving adaptation module to ", path)
        torch.save(self.adaptation_module.state_dict(), path)

    def load_adaptation_module(self, path):
        print("Loading adaptation module from", path)
        self.adaptation_module.load_state_dict(torch.load(path))

    def save_encoder(self):
        path = os.path.join(self.save_path, "encoder.pth")
        print("Saving encoder to ", path)
        torch.save(self.rma_encoder.state_dict(), path)

    def load_encoder(self, path):
        print("Loading encoder from", path)
        self.rma_encoder.load_state_dict(torch.load(path))

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

# # DEBUG
# if __name__ == "__main__":
#     robot_saved_obs = pickle.load(open("../../robot_computed_obs.pkl", "rb"))
#     isaac_saved_obs = pickle.load(open("../../isaac_saved_obs.pkl", "rb"))
#     saved_obs = robot_saved_obs
#     buffer_size = 50

#     rma = RMA([256, 128, 18], 22, 50, 56, "cpu", "/tmp")
#     rma.load_adaptation_module("../../adaptation_module.pth")
#     rma.load_encoder("../../encoder.pth")
#     adaptation_module_onnx = OnnxInfer(
#         "../../adaptation_module.onnx", "rma_history", awd=True
#     )

#     onnx_latents = []
#     torch_latents = []

#     buffer = np.zeros((buffer_size, 56), dtype=np.float32)
#     for obs in saved_obs:
#         buffer = np.roll(buffer, 1, axis=0)
#         buffer[0] = obs

#         buffer_torch = torch.from_numpy(buffer)
#         onnx_latent = adaptation_module_onnx.infer(buffer.flatten())
#         torch_latent = rma.adaptation_module(buffer_torch.flatten()).detach().numpy()

#         onnx_latents.append(onnx_latent)
#         torch_latents.append(torch_latent)

#     onnx_latents = np.array(onnx_latents)
#     torch_latents = np.array(torch_latents)

#     # plt.plot(onnx_latents, label="onnx")
#     plt.ylim(-5, 5)
#     plt.plot(torch_latents, label="torch")
#     plt.legend()
#     plt.title("robot latent")
#     plt.show()
