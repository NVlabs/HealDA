# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import math


class PositionalEmbedding(torch.nn.Module):
    """Timestep embedding used in the DDPM++ and ADM architectures.


    f = (1/M)^(i / N)
    [cos(f_i x), sin(f_i x)] for i =0,...,N - 1

    sup wavelength = sup  2 pi / f = 2 pi 1 / inf f = 2 pi / (1 / M) = 2 pi M

    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """Timestep embedding used in the NCSN++ architecture."""

    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * math.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FrequencyEmbedding(torch.nn.Module):
    """Periodic Embedding.

    Useful for inputs defined on the circle [0, 2pi)
    """

    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x):
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(torch.nn.Module):
    """Time embedding assuming 365.25 day years

    Args:
        day_of_year: (n, t)
        second_of_day: (n, t)
    Returns:
        (n, embed_channels * 4, t, x)

    """

    def __init__(self, lon, embed_channels: int, include_legacy_bug: bool = False):
        """
        Args:
            include_legacy_bug: Provided for backwards compatibility
                with existing checkpoints. If True, use the incorrect formula
                for local_time (hour - lon) instead of the correct formula (hour + lon)
        """
        super().__init__()
        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 4
        self.include_legacy_bug = include_legacy_bug

    def forward(self, day_of_year, second_of_day):
        if second_of_day.shape != day_of_year.shape:
            raise ValueError()

        if self.include_legacy_bug:
            local_time = (second_of_day.unsqueeze(2) - self.lon * 86400 // 360) % 86400
        else:
            local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400

        a = self.embed_second(local_time / 86400)
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        a, b = torch.broadcast_tensors(a, b)
        return torch.concat([a, b], dim=1)  # (n c x)


class EmbedNoiseLabels(torch.nn.Module):
    def __init__(
        self,
        emb_channels,
        label_dim,
        noise_channels,
        label_dropout=None,
        legacy_label_bias: bool = False,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)

        # legacy_label_bias: for loading old checkpoints that had Linear(0, noise_channels)
        # which contributed a trained bias even with label_dim=0
        self.map_label = None
        if label_dim != 0 or legacy_label_bias:
            self.map_label = torch.nn.Linear(label_dim, noise_channels)

        self.map_layer0 = torch.nn.Linear(
            in_features=noise_channels, out_features=emb_channels
        )
        self.map_layer1 = torch.nn.Linear(
            in_features=emb_channels, out_features=emb_channels
        )

    def forward(self, noise_labels, class_labels):
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([noise_labels.shape[0], 1], device=tmp.device)
                    >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * math.sqrt(self.map_label.in_features))

        emb = torch.nn.functional.silu(self.map_layer0(emb))
        emb = torch.nn.functional.silu(self.map_layer1(emb))
        return emb


def sphere_position(lat, lon):
    """Unit-sphere xyz."""
    lat_rad, lon_rad = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack(
        [
            torch.cos(lat_rad) * torch.cos(lon_rad),
            torch.cos(lat_rad) * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ]
    )


@torch.compiler.disable
def _solar_ephemeris(timestamp):
    """``(declination, phase)`` per ``(b, t, 1)``, in radians. No spatial extent.

    Kept out of any compiled graph because ``_obliquity_star`` calls ``np.deg2rad``, so the
    break lands on two scalars here rather than around the full field.
    """
    from physicsnemo.utils.zenith_angle import (
        _local_mean_sidereal_time,
        _right_ascension_declination,
        _timestamp_to_julian_century,
    )

    jc = _timestamp_to_julian_century(timestamp.unsqueeze(-1).to(torch.float64))
    ra, declination = _right_ascension_declination(jc)
    # h_angle = gmst(jc) + lon - ra, and only `lon` has spatial extent, so everything else
    # collapses into one phase per (b, t).
    return declination, _local_mean_sidereal_time(jc, 0.0) - ra


def solar_zenith_cosine(timestamp, lat, lon):
    """Cosine of the solar zenith angle, unclamped. (b, 1, t, npix).

    Uses physicsnemo's ephemeris, but the spatial half is written out here so it is ordinary
    torch and fuses into one kernel; calling physicsnemo's
    `cos_zenith_angle_from_timestamp` instead is 7.5x slower under torch.compile. float64
    throughout, as upstream does: the sidereal polynomials overflow float32's precision.
    """
    declination, phase = _solar_ephemeris(timestamp)
    # The ephemeris follows the timestamp, which arrives on the host; the field follows lat.
    declination = torch.as_tensor(declination, dtype=torch.float64, device=lat.device)
    phase = torch.as_tensor(phase, dtype=torch.float64, device=lat.device)
    lat_rad = torch.deg2rad(lat.to(torch.float64))
    lon_rad = torch.deg2rad(lon.to(torch.float64))
    cosine = torch.sin(lat_rad) * torch.sin(declination) + torch.cos(
        lat_rad
    ) * torch.cos(declination) * torch.cos(phase + lon_rad)
    return cosine.to(lat.dtype).unsqueeze(1)
