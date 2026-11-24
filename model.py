"""잡음 제거 모델 정의 (학습 노트북 3.2절과 동일한 구조).

저장된 가중치를 불러오려면 학습 당시와 같은 구조·설정이어야 합니다.
"""
import torch
from torch import nn

SAMPLE_RATE = 16_000
N_FFT = 512        # STFT 창 길이
HOP = 128          # STFT 이동 간격


class CRMBiLSTM(nn.Module):
    """압축 크기 특징 → Conv1d → BiLSTM → 유계 복소 마스크(cRM)."""
    def __init__(self, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.n_fft, self.hop = N_FFT, HOP
        bins = N_FFT // 2 + 1
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.norm = nn.LayerNorm(bins)
        self.pre = nn.Sequential(nn.Conv1d(bins, hidden, 3, padding=1), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, 2 * bins)

    def spectrum(self, wave):
        return torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True)

    def forward(self, noisy):
        Y = self.spectrum(noisy)                                   # [B, F, T] 복소
        feat = Y.abs().clamp_min(1e-8) ** 0.3
        feat = self.norm(feat.transpose(1, 2))                     # [B, T, F]
        h = self.pre(feat.transpose(1, 2)).transpose(1, 2)         # [B, T, H]
        h, _ = self.rnn(h)
        mr, mi = self.out(h).transpose(1, 2).chunk(2, dim=1)       # 각 [B, F, T]
        r = torch.sqrt(mr ** 2 + mi ** 2 + 1e-8)
        scale = torch.tanh(r) / r                                  # |M| ≤ 1
        return torch.complex(mr * scale, mi * scale) * Y

    def enhance(self, noisy):
        return torch.istft(self(noisy), n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=noisy.shape[-1])


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                         nn.BatchNorm2d(cout), nn.PReLU(cout))


def deconv_block(cin, cout, last=False):
    layers = [nn.ConvTranspose2d(cin, cout, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))]
    if not last:
        layers += [nn.BatchNorm2d(cout), nn.PReLU(cout)]
    return nn.Sequential(*layers)


class DPRNNBlock(nn.Module):
    """주파수 방향(intra)과 시간 방향(inter)을 번갈아 처리하는 이중 경로 RNN 블록."""
    def __init__(self, channels, hidden):
        super().__init__()
        self.intra_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(2 * hidden, channels)
        self.intra_norm = nn.LayerNorm(channels)
        self.inter_rnn = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.inter_fc = nn.Linear(2 * hidden, channels)
        self.inter_norm = nn.LayerNorm(channels)

    def forward(self, x):                                       # x: [B, C, T, F]
        b, c, t, f = x.shape
        intra = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        intra = self.intra_norm(self.intra_fc(self.intra_rnn(intra)[0]))
        x = x + intra.reshape(b, t, f, c).permute(0, 3, 1, 2)
        inter = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        inter = self.inter_norm(self.inter_fc(self.inter_rnn(inter)[0]))
        x = x + inter.reshape(b, f, t, c).permute(0, 3, 2, 1)
        return x


class DPCRN(nn.Module):
    def __init__(self, channels=(32, 64, 64), rnn_hidden=64, num_blocks=2):
        super().__init__()
        self.n_fft, self.hop = N_FFT, HOP
        c1, c2, c3 = channels
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.enc1 = conv_block(3, c1)               # F 257 → 129
        self.enc2 = conv_block(c1, c2)              # F 129 → 65
        self.enc3 = conv_block(c2, c3)              # F 65 → 33
        self.dprnn = nn.Sequential(*[DPRNNBlock(c3, rnn_hidden) for _ in range(num_blocks)])
        self.dec3 = deconv_block(2 * c3, c2)        # F 33 → 65
        self.dec2 = deconv_block(2 * c2, c1)        # F 65 → 129
        self.dec1 = deconv_block(2 * c1, 2, last=True)

    def spectrum(self, wave):
        return torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True)

    def forward(self, noisy):
        Y = self.spectrum(noisy)
        mag = Y.abs().clamp_min(1e-8)
        comp = mag ** 0.3
        feat = torch.stack([comp, Y.real * comp / mag, Y.imag * comp / mag], dim=1)
        x = feat.transpose(2, 3)                                    # [B, 3, T, F]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        h = self.dprnn(e3)
        d3 = self.dec3(torch.cat([h, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        m = self.dec1(torch.cat([d2, e1], dim=1)).transpose(2, 3)   # [B, 2, F, T]
        mr, mi = m[:, 0], m[:, 1]
        r = torch.sqrt(mr ** 2 + mi ** 2 + 1e-8)
        scale = torch.tanh(r) / r
        return torch.complex(mr * scale, mi * scale) * Y

    def enhance(self, noisy):
        return torch.istft(self(noisy), n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=noisy.shape[-1])


class DPCRNLarge(DPCRN):
    """DPRNN 블록 3개, 병목 채널 96, RNN hidden 96."""
    def __init__(self):
        super().__init__(channels=(32, 64, 96), rnn_hidden=96, num_blocks=3)


# weights/ensemble.json의 "model" 이름 → 클래스
MODEL_CLASSES = {"CRMBiLSTM": CRMBiLSTM, "DPCRN": DPCRN, "DPCRNLarge": DPCRNLarge}
