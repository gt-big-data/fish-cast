from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class AutoregressiveKrillForecaster(nn.Module):
    def __init__(self, env_dim: int, hidden_dim: int = 64, horizon_len: int = 3) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.horizon_len = horizon_len
        self.encoder = nn.GRU(input_size=env_dim + 2, hidden_size=hidden_dim, batch_first=True)
        self.decoder_cell = nn.GRUCell(input_size=env_dim + 2, hidden_size=hidden_dim)
        self.target_head = nn.Linear(hidden_dim, 1)
        self.hotspot_head = nn.Linear(hidden_dim, 1)

    def forward(self, past_inputs: torch.Tensor, future_env: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, hidden = self.encoder(past_inputs)
        hidden = hidden.squeeze(0)

        previous_target = past_inputs[:, -1, self.env_dim : self.env_dim + 1]
        previous_hotspot = past_inputs[:, -1, self.env_dim + 1 :]

        predicted_targets = []
        predicted_logits = []
        for step in range(future_env.shape[1]):
            decoder_input = torch.cat([future_env[:, step, :], previous_target, previous_hotspot], dim=1)
            hidden = self.decoder_cell(decoder_input, hidden)
            target = self.target_head(hidden)
            hotspot_logits = self.hotspot_head(hidden)
            predicted_targets.append(target)
            predicted_logits.append(hotspot_logits)
            previous_target = target
            previous_hotspot = torch.sigmoid(hotspot_logits)

        target_tensor = torch.stack(predicted_targets, dim=1).squeeze(-1)
        hotspot_logits_tensor = torch.stack(predicted_logits, dim=1).squeeze(-1)
        return {
            "density": target_tensor,
            "hotspot_logits": hotspot_logits_tensor,
            "hotspot_probability": torch.sigmoid(hotspot_logits_tensor),
        }


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        combined = torch.cat([x, hidden], dim=1)
        gates = self.gates(combined)
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gates, 4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def init_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        shape = (batch_size, self.hidden_dim, height, width)
        hidden = torch.zeros(shape, device=device, dtype=dtype)
        cell = torch.zeros(shape, device=device, dtype=dtype)
        return hidden, cell


class ConvLSTMKrillForecaster(nn.Module):
    def __init__(self, env_dim: int, hidden_dim: int = 32, horizon_len: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.horizon_len = horizon_len
        self.encoder_cell = ConvLSTMCell(input_dim=env_dim + 2, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.decoder_cell = ConvLSTMCell(input_dim=env_dim + 2, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.target_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.hotspot_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, past_inputs: torch.Tensor, future_env: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _, _, height, width = past_inputs.shape
        state = self.encoder_cell.init_state(
            batch_size=batch_size,
            spatial_size=(height, width),
            device=past_inputs.device,
            dtype=past_inputs.dtype,
        )
        for step in range(past_inputs.shape[1]):
            state = self.encoder_cell(past_inputs[:, step], state)

        hidden, cell = state
        previous_target = past_inputs[:, -1, self.env_dim : self.env_dim + 1]
        previous_hotspot = past_inputs[:, -1, self.env_dim + 1 : self.env_dim + 2]

        predicted_targets = []
        predicted_logits = []
        for step in range(future_env.shape[1]):
            decoder_input = torch.cat([future_env[:, step], previous_target, previous_hotspot], dim=1)
            hidden, cell = self.decoder_cell(decoder_input, (hidden, cell))
            target = self.target_head(hidden)
            hotspot_logits = self.hotspot_head(hidden)
            predicted_targets.append(target)
            predicted_logits.append(hotspot_logits)
            previous_target = target
            previous_hotspot = torch.sigmoid(hotspot_logits)

        target_tensor = torch.stack(predicted_targets, dim=1)
        hotspot_logits_tensor = torch.stack(predicted_logits, dim=1)
        return {
            "density": target_tensor,
            "hotspot_logits": hotspot_logits_tensor,
            "hotspot_probability": torch.sigmoid(hotspot_logits_tensor),
        }
