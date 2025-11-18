import torch
import torch.nn as nn

# %% 7) CNN–LSTM model
def conv_block(in_ch, out_ch, k, p):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True)
    )


class ParallelCNN_LSTM(nn.Module):
    """
    Parallel CNN (Tower A: 3x Conv1D, k=3, 64ch)  ||
                 (Tower B: 3x Conv1D, k=5, 128ch -> LSTM(H) -> LSTM(2H) -> BN -> Dense(td))
    -> concat(A, B') -> Global Average Pool (time) -> Dense(n_classes)
    Input: (B, 1, T) with optional trailing singleton dim (B,1,T,1)
    """
    def __init__(self, n_classes=4, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()
        H1 = lstm_hidden               # e.g., 128
        H2 = 2 * lstm_hidden           # e.g., 256

        # ----- Tower A: CNN only -----
        self.a1 = conv_block(1, 64, k=3, p=1)
        self.a2 = conv_block(64, 64, k=3, p=1)
        self.a3 = conv_block(64, 64, k=3, p=1)

        # ----- Tower B: CNN -> LSTM(H1) -> LSTM(H2) -> BN -> Dense(td) -----
        self.b1 = conv_block(1, 128, k=5, p=2)
        self.b2 = conv_block(128, 128, k=5, p=2)
        self.b3 = conv_block(128, 128, k=5, p=2)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=H1, num_layers=1,
                             batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=H1, hidden_size=H2, num_layers=1,
                             batch_first=True, bidirectional=False)

        self.bn_b = nn.BatchNorm1d(H2)

        # Time-distributed Dense to project B path (per-timestep Linear)
        self.time_dense_b = nn.Linear(H2, 128)

        # Final head after concatenation
        self.fc = nn.Linear(64 + 128, n_classes)

    def forward(self, x):
        # x: (B, 1, T) or (B,1,T,1)
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)

        # ----- Tower A -----
        xa = self.a3(self.a2(self.a1(x)))                  # (B, 64, T)

        # ----- Tower B -----
        xb = self.b3(self.b2(self.b1(x)))                  # (B, 128, T)
        xb_t = xb.transpose(1, 2)                          # (B, T, 128)

        xb_t, _ = self.lstm1(xb_t)                         # (B, T, H1)
        xb_t, _ = self.lstm2(xb_t)                         # (B, T, H2)

        xb_c = xb_t.transpose(1, 2)                        # (B, H2, T)
        xb_c = self.bn_b(xb_c)

        xb_t = xb_c.transpose(1, 2)                        # (B, T, H2)
        xb_t = self.time_dense_b(xb_t)                     # (B, T, 128)
        xb_proj = xb_t.transpose(1, 2)                     # (B, 128, T)

        # ----- Concatenate & Head -----
        z = torch.cat([xa, xb_proj], dim=1)                # (B, 64+128, T)
        z = z.mean(dim=2)                                  # GAP over time -> (B, 192)
        logits = self.fc(z)                                # (B, n_classes)
        return logits
