import torch
import torch.nn as nn


class FFTMixing(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fft(x, dim=1)
        x = torch.fft.fft(x, dim=2)
        return self.dropout(x.real)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()

        self._norm1 = nn.LayerNorm(embedding_dim)
        self._fft = FFTMixing(dropout)
        self._norm2 = nn.LayerNorm(embedding_dim)
        self._ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        _mask = mask.unsqueeze(-1).to(x.dtype)
        residual = x
        x = self._norm1(x)
        x = x * _mask
        x = self._fft(x)
        x = residual + x
        x = x * _mask

        residual = x
        x = self._norm2(x)
        x = self._ffn(x)
        x = residual + x
        x = x * _mask

        return x


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int, dropout: float):
        super(Encoder, self).__init__()
        self._layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, dropout) for _ in range(num_layers)]
        )

        self._norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x, mask)
        return self._norm(x)


class Model(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super(Model, self).__init__()
        self._segment_embedding = nn.Embedding(2, embedding_dim)
        self._encoder = Encoder(embedding_dim, num_layers, dropout)
        self._classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / count

    def forward(
        self,
        x: torch.FloatTensor,
        x_mask: torch.BoolTensor,
        y: torch.FloatTensor,
        y_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        batch_size = x.size(0)
        device = x.device
        _input = torch.cat([x, y], dim=1)
        x_segment = torch.zeros(batch_size, x.size(1), dtype=torch.long, device=device)
        y_segment = torch.ones(batch_size, y.size(1), dtype=torch.long, device=device)
        segment_ids = torch.cat([x_segment, y_segment], dim=1)
        _input = _input + self._segment_embedding(segment_ids)
        mask = torch.cat([x_mask, y_mask], dim=1).bool()
        _output = self._encoder(_input, mask)
        Lp = x.size(1)
        u = self.masked_mean_pool(_output[:, :Lp], x_mask)
        v = self.masked_mean_pool(_output[:, Lp:], y_mask)
        features = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        return self._classifier(features)
