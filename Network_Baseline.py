import torch
import torch.nn as nn

class Dave2Regression(nn.Module):
    def __init__(self, dropout: float = 0.2, out_dim: int = 3):
        """
        Image-only DAVE-2 style regression network.
        Input: (B,3,66,200) YUV normalized to [-1,1]
        Output:
          out_dim=1 -> steer in [-1,1]
          out_dim=3 -> [throttle, steer, brake] with correct ranges
        """
        super().__init__()
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(inplace=True),
        )

        # compute flattened conv size dynamically for (3,66,200)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            conv_out = self.conv(dummy)
            self._flat_dim = conv_out.numel()  # e.g., 1152

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self._flat_dim, 100), nn.ELU(inplace=True),
            nn.Linear(100, 50),             nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(50, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        out = self.fc(x)  # raw

        if self.out_dim == 1:
            # steer only in [-1,1]
            return torch.tanh(out)

        if self.out_dim == 3:
            # [throttle, steer, brake] in [0,1],[-1,1],[0,1]
            thr   = torch.sigmoid(out[:, 0:1])
            steer = torch.tanh(out[:, 1:2])
            brk   = torch.sigmoid(out[:, 2:3])
            return torch.cat([thr, steer, brk], dim=1)

        return out


    