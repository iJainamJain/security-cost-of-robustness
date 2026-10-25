"""Transport-channel simulation.

A "channel" is whatever a delivery path does to a carrier between sender and
receiver: recompression, resizing, chroma subsampling, format conversion. In the
robust-steganography literature the channel is almost always a single JPEG
quality factor, which is why published methods need to know it in advance. The
RRS proposal (RESEARCH_DOSSIER.md Part V, layer L2) instead optimises against a
*distribution* over channels, so this module is built around composing and
sampling channels rather than applying one fixed operation.

All channels are ``Channel`` instances: callables from a PIL image to a PIL
image, carrying a stable ``name`` used as a results key.
"""
from __future__ import annotations

import dataclasses
import io
import random
from typing import Callable, Iterable, List, Sequence

from PIL import Image

__all__ = [
    "Channel",
    "identity",
    "jpeg",
    "resize",
    "subsample",
    "webp",
    "compose",
    "STANDARD_JPEG_SWEEP",
    "PLATFORM_LIKE",
    "ChannelEnsemble",
]


@dataclasses.dataclass(frozen=True)
class Channel:
    """A named, deterministic image-to-image transformation."""

    name: str
    fn: Callable[[Image.Image], Image.Image]

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.fn(img)


def _roundtrip(img: Image.Image, fmt: str, **save_kwargs) -> Image.Image:
    """Encode to an in-memory buffer and decode back.

    Going through a buffer rather than a temp file keeps this fast enough
    (~0.5 ms for a 512x512 JPEG) that sweeping thousands of images is free.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # force decode before the buffer goes out of scope
    return out


def identity() -> Channel:
    return Channel("identity", lambda im: im.copy())


def jpeg(quality: int, subsampling: str | int = "keep") -> Channel:
    """JPEG recompression at a given quality factor.

    ``subsampling="keep"`` preserves the source's chroma subsampling, which is
    what most platforms do to already-JPEG uploads. Pass 0 (4:4:4) or 2 (4:2:0)
    to force it.
    """
    def fn(im: Image.Image) -> Image.Image:
        # "keep" is only legal when the source is itself a JPEG.
        sub = subsampling
        if sub == "keep" and im.format != "JPEG":
            sub = 2
        return _roundtrip(im, "JPEG", quality=quality, subsampling=sub)

    tag = f"jpeg{quality}" + ("" if subsampling == "keep" else f"_s{subsampling}")
    return Channel(tag, fn)


def resize(scale: float, resample: int = Image.BILINEAR) -> Channel:
    """Downscale by ``scale`` then back to the original size.

    This is the operation that defeats most spread-spectrum schemes, because it
    resamples the coefficient grid the payload is aligned to.
    """
    def fn(im: Image.Image) -> Image.Image:
        w, h = im.size
        small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)
        return small.resize((w, h), resample)

    return Channel(f"resize{scale:g}", fn)


def subsample(mode: int = 2) -> Channel:
    """Force chroma subsampling via a high-quality JPEG round-trip."""
    return Channel(f"subsample{mode}",
                   lambda im: _roundtrip(im, "JPEG", quality=98, subsampling=mode))


def webp(quality: int = 80) -> Channel:
    """WebP conversion — what several platforms now do to uploaded images."""
    return Channel(f"webp{quality}",
                   lambda im: _roundtrip(im, "WEBP", quality=quality).convert("RGB"))


def compose(*channels: Channel, name: str | None = None) -> Channel:
    """Chain channels left to right."""
    chained = tuple(channels)
    label = name or "+".join(c.name for c in chained)

    def fn(im: Image.Image) -> Image.Image:
        for c in chained:
            im = c(im)
        return im

    return Channel(label, fn)


# The literature's standard axis: cover at one QF, channel at another.
STANDARD_JPEG_SWEEP: Sequence[Channel] = tuple(
    jpeg(q) for q in (95, 90, 85, 75, 65, 50)
)

# Stand-ins for real platform pipelines. These are *guesses* until measured —
# PLAN.md M2 calls for characterising the real ones. Do not report these as
# platform behaviour without measuring first.
PLATFORM_LIKE: Sequence[Channel] = (
    compose(jpeg(85), name="platform_mild"),
    compose(resize(0.75), jpeg(80), name="platform_resize_recompress"),
    compose(jpeg(75), subsample(2), name="platform_aggressive"),
    compose(webp(80), name="platform_webp"),
)


class ChannelEnsemble:
    """A distribution over channels, for distributionally-robust embedding.

    Existing robust schemes optimise against one known channel. Sampling from an
    ensemble is what lets a scheme be channel-blind (dossier Part V, L2).
    """

    def __init__(self, channels: Iterable[Channel], weights: Sequence[float] | None = None):
        self.channels: List[Channel] = list(channels)
        if not self.channels:
            raise ValueError("ChannelEnsemble needs at least one channel")
        if weights is None:
            self.weights = [1.0] * len(self.channels)
        else:
            if len(weights) != len(self.channels):
                raise ValueError("weights and channels differ in length")
            self.weights = list(weights)

    def sample(self, rng: random.Random | None = None) -> Channel:
        r = rng or random
        return r.choices(self.channels, weights=self.weights, k=1)[0]

    def apply_all(self, img: Image.Image) -> dict[str, Image.Image]:
        """Every channel applied to one image, keyed by channel name."""
        return {c.name: c(img) for c in self.channels}

    def __len__(self) -> int:
        return len(self.channels)

    def __repr__(self) -> str:
        return f"ChannelEnsemble({[c.name for c in self.channels]})"


DEFAULT_ENSEMBLE = ChannelEnsemble(list(STANDARD_JPEG_SWEEP) + list(PLATFORM_LIKE))
