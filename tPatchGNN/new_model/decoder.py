import torch.nn as nn

class ForecastDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.hid_dim + args.te_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, 1)
        )

    def forward(self, h, te_pred):
        return self.decoder(torch.cat([h, te_pred], dim=-1))
