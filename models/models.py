import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_classes, num_heads=8, num_layers=4, dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),  # normalize input per time step
            nn.Linear(input_dim, model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)  # (B, T, D)
        x = self.transformer(x)  # (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)
        x = self.dropout(x)
        return self.classifier(x)  # (B, num_classes)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, projection_dim=64):
        super().__init__()
        # Base encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

        # Optional projection head (often used in contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x, project=True):
        x = self.encoder(x)
        if project:
            x = self.projector(x)
        return x


class TimeSeriesGCMSNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq, hidden)
        pooled = torch.mean(lstm_out, dim=1)  # average over time
        embedding = self.embedding_layer(pooled)
        out = self.classifier(embedding)
        return out, embedding