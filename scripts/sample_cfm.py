import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict
from scipy.integrate import solve_ivp


class ImageEncoder(nn.Module):
    """Simple CNN encoder for visual observations"""

    def __init__(self, input_channels: int = 3, output_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fc = nn.Linear(64 * 7 * 7, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RobotStateEncoder(nn.Module):
    """MLP encoder for robot proprioceptive state"""

    def __init__(self, state_dim: int, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Conditional1DUNet(nn.Module):
    """1D U-Net for velocity field prediction in CFM"""

    def __init__(self, action_dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.condition_dim = condition_dim

        # Time embedding
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)

        # 1D U-Net architecture
        self.down_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(action_dim + 64, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(8, hidden_dim * 2),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim * 2,
                        hidden_dim * 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(8, hidden_dim * 4),
                    nn.SiLU(),
                ),
            ]
        )

        # Middle block
        self.middle = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.SiLU(),
        )

        # Up layers
        self.up_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dim * 8,
                        hidden_dim * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(8, hidden_dim * 2),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dim * 4, hidden_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                ),
                nn.Conv1d(hidden_dim * 2, action_dim, kernel_size=3, padding=1),
            ]
        )

        # Condition fusion layers
        self.condition_fusion = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.Linear(hidden_dim, hidden_dim * 4),
            ]
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Action sequence [batch_size, action_dim, sequence_length]
            t: Time [batch_size, 1]
            condition: Combined condition [batch_size, condition_dim]
        """
        batch_size, _, seq_len = x.shape

        # Time embedding
        t_emb = self.time_mlp(t)  # [batch_size, 64]
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, seq_len)  # [batch_size, 64, seq_len]

        # Condition embedding
        cond_emb = self.condition_proj(condition)  # [batch_size, hidden_dim]

        # Concatenate action and time
        x = torch.cat([x, t_emb], dim=1)  # [batch_size, action_dim + 64, seq_len]

        # Down path
        skip_connections = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            # Fuse condition
            if i < len(self.condition_fusion):
                cond_proj = self.condition_fusion[i](cond_emb).unsqueeze(-1)
                x = x + cond_proj

            skip_connections.append(x)

        # Middle
        x = self.middle(x)

        # Up path
        for i, layer in enumerate(self.up_layers[:-1]):
            skip = skip_connections[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        # Final layer
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.up_layers[-1](x)

        return x


class CFMRobotPolicy(nn.Module):
    """Complete CFM policy for robot action generation"""

    def __init__(self, action_dim: int, state_dim: int, sequence_length: int = 16):
        super().__init__()
        self.action_dim = action_dim
        self.sequence_length = sequence_length

        # Encoders
        self.image_encoder = ImageEncoder(output_dim=256)
        self.state_encoder = RobotStateEncoder(state_dim, output_dim=128)

        # Velocity field network
        condition_dim = 256 + 128  # image + state
        self.velocity_net = Conditional1DUNet(action_dim, condition_dim)

    def encode_condition(
        self, image: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Encode image and robot state into condition vector"""
        img_feat = self.image_encoder(image)  # [batch_size, 256]
        state_feat = self.state_encoder(state)  # [batch_size, 128]
        return torch.cat([img_feat, state_feat], dim=-1)  # [batch_size, 384]

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field"""
        return self.velocity_net(x_t, t, condition)


class RobotDataset(Dataset):
    """Dataset for robot demonstrations"""

    def __init__(self, images: np.ndarray, states: np.ndarray, actions: np.ndarray):
        self.images = torch.FloatTensor(images)
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "state": self.states[idx],
            "action": self.actions[idx],
        }


def cfm_loss(model: CFMRobotPolicy, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """CFM training loss"""
    batch_size = batch["action"].shape[0]
    device = batch["action"].device

    # Sample time uniformly
    t = torch.rand(batch_size, 1, device=device)

    # Sample noise (x_0)
    x_0 = torch.randn_like(batch["action"])

    # Target action (x_1)
    x_1 = batch["action"]

    # Interpolate: x_t = (1-t) * x_0 + t * x_1
    t_expand = t.unsqueeze(-1).expand_as(x_1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Target velocity: v_t = x_1 - x_0
    v_target = x_1 - x_0

    # Encode condition
    condition = model.encode_condition(batch["image"], batch["state"])

    # Predict velocity
    v_pred = model(x_t.transpose(1, 2), t, condition)
    v_pred = v_pred.transpose(1, 2)

    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    return loss


def sample_actions(
    model: CFMRobotPolicy, image: torch.Tensor, state: torch.Tensor, num_steps: int = 50
) -> torch.Tensor:
    """Sample actions using ODE solver"""
    model.eval()
    device = next(model.parameters()).device
    batch_size = image.shape[0]

    # Encode condition
    condition = model.encode_condition(image, state)

    # Initial noise
    x_0 = torch.randn(
        batch_size, model.sequence_length, model.action_dim, device=device
    )

    def velocity_fn(t, x_flat):
        """Velocity function for ODE solver"""
        x = (
            torch.FloatTensor(x_flat)
            .reshape(batch_size, model.sequence_length, model.action_dim)
            .to(device)
        )
        t_tensor = torch.full((batch_size, 1), t, device=device)

        with torch.no_grad():
            v = model(x.transpose(1, 2), t_tensor, condition)
            v = v.transpose(1, 2)

        return v.cpu().numpy().flatten()

    # Solve ODE from t=0 to t=1
    t_span = [0, 1]
    t_eval = np.linspace(0, 1, num_steps)

    result = solve_ivp(
        velocity_fn, t_span, x_0.cpu().numpy().flatten(), t_eval=t_eval, method="dopri5"
    )

    # Return final actions
    final_actions = result.y[:, -1].reshape(
        batch_size, model.sequence_length, model.action_dim
    )
    return torch.FloatTensor(final_actions)


def train_cfm_policy():
    """Training loop example"""
    # Hyperparameters
    action_dim = 7  # e.g., 7-DOF robot arm
    state_dim = 14  # joint positions + velocities
    sequence_length = 16
    batch_size = 128
    learning_rate = 1e-6
    num_epochs = 100

    # Initialize model
    model = CFMRobotPolicy(action_dim, state_dim, sequence_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Dummy data (replace with real robot demonstrations)
    images = np.random.randn(1000, 3, 64, 64)
    states = np.random.randn(1000, state_dim)
    actions = np.random.randn(1000, sequence_length, action_dim)

    dataset = RobotDataset(images, states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            loss = cfm_loss(model, batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model


# Usage example
if __name__ == "__main__":
    # Train model
    trained_model = train_cfm_policy()

    # Generate actions for new observation
    dummy_image = torch.randn(1, 3, 64, 64)
    dummy_state = torch.randn(1, 14)

    generated_actions = sample_actions(trained_model, dummy_image, dummy_state)
    print(f"Generated action sequence shape: {generated_actions.shape}")
