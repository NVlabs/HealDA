# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import earth2grid
import earth2grid.healpix
import einops
import torch
import dataclasses

from healda.observations.types import healpix_pixel_order
from healda.models.embedding import CalendarEmbedding


@dataclasses.dataclass
class Subdomain:
    """Specification of the subdomain


    Attrs:
        x: (b, nf) the x coordinate of the lower left corner
        y: (b, nf) the y coordinate of the lower left corner
        f: (b, nf) the face of the lower left corner
        n: the size of the subdomain

    describes a nf * n * n shaped tensor.

    For example::

        Subdomain(x=torch.zeros([1, 12]), y=torch.zeros([1, 12]), f=torch.arange(12).unsqueeze(0))

    describes the full healpix domain.

    """

    x: torch.Tensor
    y: torch.Tensor
    f: torch.Tensor
    n: int
    pixel_order = earth2grid.healpix.HEALPIX_PAD_XY
    level: int

    def __post_init__(self):
        is_valid = self.x.shape == self.y.shape == self.f.shape
        if not is_valid:
            raise ValueError("all attributes must have same shape")

        if self.level < 0:
            raise ValueError("level must be nonnegative")

    @property
    def num_faces(self):
        return self.x.shape[-1]

    def select_from_global(self, global_: torch.Tensor):
        """Select subdomain pixels from a global HEALPix tensor.

        Args:
            global_: Shape (b, c, t, npix_global)

        Returns:
            Shape (b, c, t, npix_subdomain)
        """
        i = torch.arange(self.n, device=global_.device)
        xi = self.x.unsqueeze(2).unsqueeze(3) + i
        yi = self.y.unsqueeze(2).unsqueeze(3) + i.unsqueeze(1)
        f = self.f.unsqueeze(2).unsqueeze(3)

        nside = 2**self.level
        pix = f * nside * nside + yi * nside + xi  # (b, f, n, n)
        pix = einops.rearrange(pix, "b f x y -> b () () (f x y)")
        shape = (*global_.shape[:-1], pix.shape[-1])
        pix = pix.expand(shape)
        return global_.gather(-1, pix)

    def coarsen(self, levels: int) -> "Subdomain":
        factor = 2**levels
        return Subdomain(
            self.x // factor,
            self.y // factor,
            self.f,
            self.n // factor,
            level=self.level - levels,
        )


class HPXPatchEmbed(torch.nn.Module):
    """

    Args:
        input: (b c t x)
    Returns
        output: (b t x_model c_model)

    ``token_order`` is the order the tokens are stored in. It selects which fine pixels form a
    patch and where pos_embed/calendar land, so weights do not transfer between orders.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        level_fine,
        level_coarse: int,
        use_gains: bool = False,
        allow_nans: bool = False,
        token_order: str = "hpxpadxy",
        # False for wrappers that add their own calendar at a finer grid; adding it here too
        # would condition on time twice.
        add_calendar: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size = 2 ** (level_fine - level_coarse)
        self.allow_nans = allow_nans
        self.level_coarse = level_coarse
        self.level_fine = level_fine
        self.token_order = token_order
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.side_fine = 2**level_fine
        npix = 12 * 4**level_coarse
        # TODO adding time_length is backwards breaing
        # need to add ability to fine tune
        self.pos_embed = torch.nn.Parameter(
            torch.randn(
                npix,
                out_channels,
            )
        )
        self.calendar_embed = None
        if add_calendar:
            grid = earth2grid.healpix.Grid(
                level=level_coarse, pixel_order=healpix_pixel_order(token_order)
            )
            lon = torch.as_tensor(grid.lon)
            if out_channels % 4 != 0:
                raise ValueError()
            self.calendar_embed = CalendarEmbedding(lon, out_channels // 4).float()

        self.use_gains = use_gains
        if use_gains:
            self.pos_embed_gain = torch.nn.Parameter(torch.tensor(0.01))
            if add_calendar:
                self.calendar_embed_gain = torch.nn.Parameter(torch.tensor(0.01))

        self.null_token = None
        if self.allow_nans:
            self.null_token = torch.nn.Parameter(torch.randn(1, out_channels, 1, 1))

    def _forward_nest(self, x, *, second_of_day, day_of_year):
        # A NEST patch is patch_size**2 consecutive indices, so the conv (kernel == stride, no
        # padding) is a reshape and a matmul against the same weight, viewed.
        b, c, t, _ = x.shape
        k = self.patch_size**2
        x = einops.rearrange(x, "b c t (p k) -> (b t p) (c k)", k=k)

        num_valid = None
        num_valid_expected = float(c * k)
        if self.allow_nans:
            num_valid = num_valid_expected - x.isnan().sum(-1, keepdim=True)
            x = x.nan_to_num(0)

        x = torch.nn.functional.linear(
            x, self.conv.weight.reshape(self.conv.out_channels, -1), self.conv.bias
        )

        if num_valid is not None:
            # Clamped so the backward does not produce NaNs where num_valid is 0.
            factor = torch.sqrt_(num_valid_expected / num_valid.clamp_min(1.0))
            x = torch.where(num_valid == 0, self.null_token.reshape(1, -1), x * factor)

        x = einops.rearrange(x, "(b t p) c -> b t p c", b=b, t=t)
        return self._add_embeddings(x, second_of_day, day_of_year, None, b)

    def forward(
        self, x, *, second_of_day, day_of_year, subdomain: Subdomain | None = None
    ):
        b, _, t, _ = x.shape

        if self.token_order == "nest":
            if subdomain is not None:
                raise NotImplementedError(
                    "subdomain selection is face-based and has no nest path."
                )
            return self._forward_nest(
                x, second_of_day=second_of_day, day_of_year=day_of_year
            )

        if subdomain is None:
            f = 12
            n = self.side_fine
        else:
            f = subdomain.num_faces
            n = subdomain.n

        x = einops.rearrange(x, "b c t (f x y) -> (b t f) c x y", f=f, x=n)

        def count_nan(x, patch_size):
            x = einops.rearrange(
                x,
                "n c (cx x) (cy y) -> n () cx cy (c x y)",
                x=patch_size,
                y=patch_size,
            )
            return x.isnan().sum(-1)

        def valid_expected(x, patch_size):
            return float(x.shape[1] * patch_size * patch_size)

        num_valid = None
        num_valid_expected = valid_expected(x, self.patch_size)
        if self.allow_nans:
            # increase the convolution input by this ratio
            num_nans = count_nan(x, self.patch_size)
            num_valid = num_valid_expected - num_nans

            x = x.nan_to_num(0)

        x = self.conv(x)

        if num_valid is not None:
            # this clamping is necessary to avoid NaNs in the gradient
            # even though factors is always > 0 if num_valid is not 0, the backwards pass isn't that smart so
            # will still produce NaNs
            denom = num_valid.clamp_min(1.0)
            factor = torch.sqrt_(num_valid_expected / denom)
            x = torch.where(num_valid == 0, self.null_token, x * factor)

        x = einops.rearrange(x, "(b t f) c x y -> b t (f x y) c", b=b, t=t)
        return self._add_embeddings(x, second_of_day, day_of_year, subdomain, b)

    def _add_embeddings(self, x, second_of_day, day_of_year, subdomain, b):
        calendar_emb = None
        if self.calendar_embed is not None:
            calendar_emb = self.calendar_embed(
                second_of_day=second_of_day, day_of_year=day_of_year
            )  # b c t x

            # TODO push subdomain logic into the calendar embedding to avoid unecessary compute
            if subdomain is not None:
                coarse_subdomain = subdomain.coarsen(
                    self.level_fine - self.level_coarse
                )
                calendar_emb = coarse_subdomain.select_from_global(calendar_emb)

            calendar_emb = einops.rearrange(calendar_emb, "b c t x -> b t x c")

        pos_embed = self.pos_embed
        if subdomain is not None:
            coarse_subdomain = subdomain.coarsen(self.level_fine - self.level_coarse)
            # pos_embed is (npix, c), reshape to global format for selection
            pos_embed_global = pos_embed.unsqueeze(0).unsqueeze(2)  # (1, npix, 1, c)
            pos_embed_global = pos_embed_global.expand(b, -1, -1, -1)  # (b, npix, 1, c)
            pos_embed_global = einops.rearrange(pos_embed_global, "b x t c -> b c t x")
            pos_embed = coarse_subdomain.select_from_global(pos_embed_global)
            pos_embed = einops.rearrange(pos_embed, "b c t x -> b t x c").squeeze(0)

        if self.use_gains:
            pos_embed = self.pos_embed_gain * pos_embed
        x = x + pos_embed
        if calendar_emb is None:
            return x
        if self.use_gains:
            calendar_emb = self.calendar_embed_gain * calendar_emb
        return x + calendar_emb


class HPXPatchDecode(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        level_coarse,
        level_fine,
        token_order: str = "hpxpadxy",
    ):
        super().__init__()
        self.patch_size = patch_size = 2 ** (level_fine - level_coarse)
        self.level_in = level_coarse
        self.token_order = token_order
        self.conv_t = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _forward_nest(self, x):
        b, t, _, _ = x.shape
        # ConvTranspose2d weight is (in, out, kh, kw) and nothing slides, so each input token maps
        # to patch_size**2 consecutive NEST children by a plain matmul.
        weight = self.conv_t.weight.reshape(self.conv_t.in_channels, -1)
        y = torch.nn.functional.linear(x.reshape(-1, x.shape[-1]), weight.T)
        y = y.reshape(-1, self.conv_t.out_channels, self.patch_size**2)
        y = y + self.conv_t.bias[None, :, None]
        return einops.rearrange(y, "(b t p) c k -> b c t (p k)", b=b, t=t)

    def forward(self, x, subdomain: Subdomain | None = None):
        """Forward pass.

        Args:
            x: Input tensor of shape (b, t, npix, c)
            subdomain: Optional subdomain specification at the coarse level (level_in),
                npix = subdomain.n ^ 2 * subdomain.num_faces
        """
        if self.token_order == "nest":
            if subdomain is not None:
                raise NotImplementedError(
                    "subdomain selection is face-based and has no nest path."
                )
            return self._forward_nest(x)

        b = x.shape[0]
        f = 12
        n = 2**self.level_in
        if subdomain is not None:
            f = subdomain.num_faces
            n = subdomain.n

        x = einops.rearrange(
            x,
            "b t (f x y) c -> (b t f) c x y",
            f=f,
            x=n,
            y=n,
        )
        x = self.conv_t(x)
        x = einops.rearrange(x, "(b t f) c x y -> b c t (f x y)", f=f, b=b)
        return x
