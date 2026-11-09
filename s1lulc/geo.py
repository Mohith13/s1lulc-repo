"""Pixel-grid geometry that does not need Earth Engine.

Radar and labels must land on exactly the same 10 m grid. We use the local UTM zone (square
metre pixels) and snap the area-of-interest corner to a multiple of the pixel size, then pass
an explicit affine transform to every export. This is the 'label matching' step in the report.
"""

from __future__ import annotations

from dataclasses import dataclass


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing (lon, lat)."""
    if not -180 <= lon <= 180 or not -80 <= lat <= 84:
        raise ValueError(f"UTM is defined for -80..84 deg latitude; got lon={lon}, lat={lat}")
    zone = min(int((lon + 180) // 6) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class PixelGrid:
    """A north-up pixel grid: top-left corner (x0, y1), pixel size, width and height in pixels."""

    epsg: int
    x0: float
    y1: float
    pixel_size: float
    width: int
    height: int

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def transform(self) -> list[float]:
        """Earth Engine crsTransform: [scaleX, shearX, translateX, shearY, scaleY, translateY]."""
        return [self.pixel_size, 0.0, self.x0, 0.0, -self.pixel_size, self.y1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in the projected CRS."""
        return (
            self.x0,
            self.y1 - self.height * self.pixel_size,
            self.x0 + self.width * self.pixel_size,
            self.y1,
        )


def snap_grid(x: float, y: float, epsg: int, half_size: float, pixel_size: float) -> PixelGrid:
    """Square grid centred (to the nearest pixel) on projected point (x, y)."""
    if half_size <= 0 or pixel_size <= 0:
        raise ValueError("half_size and pixel_size must be positive")
    n = int(round(2 * half_size / pixel_size))
    x0 = round((x - half_size) / pixel_size) * pixel_size
    y1 = round((y + half_size) / pixel_size) * pixel_size
    return PixelGrid(epsg=epsg, x0=float(x0), y1=float(y1), pixel_size=float(pixel_size),
                     width=n, height=n)
