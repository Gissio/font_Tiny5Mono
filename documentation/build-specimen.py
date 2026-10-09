# Procedural specimen images for Tiny5 Mono.
#
# Tiny5 Mono's pixels are not square: each is 89 font units wide and 128
# tall, so a 6-pixel character cell is narrower than Tiny5's. The specimens
# therefore show it only where pixels are narrow too, or drawn true to
# shape: a color CRT, a character LCD, a thermal and a 9-pin printer, and
# the axes animation, which sets the font itself.
#
# Text is composed on a grid of character cells, each 6 font pixels wide and
# 9 tall, and every glyph is rasterized once into its cell from the variable
# font. The finished font-pixel image is then drawn as the device would,
# every font pixel 89:128 wide to tall.
#
# The file runs from the general to the particular: the font, the text
# screen, the shared image helpers, then one section per device. Sections
# no longer in the specimen set stay, their builders uncalled.
#
# Besides the build's own requirements, the thermal receipt is path-traced
# with Mitsuba 3 (pip install mitsuba), and the animation is encoded by
# ffmpeg, which must be on the PATH.

import math
import random
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


# --- The font ---------------------------------------------------------------

ROOT_PATH = Path(__file__).resolve().parent.parent
OUT_PATH = Path(__file__).resolve().parent

MONO_PATH = ROOT_PATH / "fonts/variable/Tiny5Mono[BLED,JITT,ROND,wdth,wght].ttf"
MONO_ITALIC_PATH = ROOT_PATH / "fonts/variable/Tiny5Mono-Italic[BLED,JITT,ROND,wdth,wght].ttf"

# Font version, as the devices display it; keep in sync with the font's
# version (name ID 5)
VERSION = "V2.007"

# The character cell, in font pixels: 6 across, and 9 down, from the
# descender line 2 pixels below the baseline to the cap line of accented
# capitals 7 pixels above it
CELL_WIDTH = 6
CELL_HEIGHT = 9
CELL_ASCENT = 7
CAP_PIXELS = 5                  # the capitals' height, above the baseline

# A font pixel, in font units at 1024 units per em
PIXEL_WIDTH = 89
PIXEL_HEIGHT = 128
PIXEL_ASPECT = PIXEL_WIDTH / PIXEL_HEIGHT


def pixel_aspect(wdth):
    """Return the width of a font pixel, of its height, at a setting of the
    Width axis: it widens the pixel in proportion, and leaves the glyphs'
    pixel grid as it is."""
    return PIXEL_ASPECT * wdth / 100

# Variation axes: weight, width, roundness, bleed, jitter. Regular, at which
# every font pixel fills its square exactly.
REGULAR_AXES = (400, 100, 0, 0, 0)

# Glyphs are rasterized at this many image pixels per font pixel, down, and
# sampled at the center of every font pixel
RASTER_SCALE = 16


@lru_cache(maxsize=None)
def get_font(axes=REGULAR_AXES, italic=False):
    """Load Tiny5 Mono at the raster size, at the given axis settings."""
    path = MONO_ITALIC_PATH if italic else MONO_PATH
    font = ImageFont.truetype(font=str(path), size=8 * RASTER_SCALE)
    font.set_variation_by_axes(list(axes))

    return font


@lru_cache(maxsize=None)
def glyph_cell(char, axes=REGULAR_AXES, italic=False):
    """Rasterize a character into its cell: a mask of CELL_WIDTH by
    CELL_HEIGHT font pixels.

    Each character is drawn on its own, at the left edge of its cell, so the
    fractional advance of the raster size never accumulates along a line,
    and ink that would spill out of the cell is clipped, as a character
    generator clips it.
    """
    font = get_font(axes, italic)
    unit_x = PIXEL_WIDTH * 8 * RASTER_SCALE / 1024
    unit_y = RASTER_SCALE
    width = math.ceil(CELL_WIDTH * unit_x)
    height = CELL_HEIGHT * unit_y
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).text(xy=(0, CELL_ASCENT * unit_y), text=char, fill=255,
                             font=font, anchor="ls")

    cell = img.resize((CELL_WIDTH, CELL_HEIGHT), Image.NEAREST,
                      box=(0, 0, CELL_WIDTH * unit_x, height))

    return cell.point(lambda v: 255 if v >= 128 else 0)


@lru_cache(maxsize=None)
def covered_codepoints():
    """Return the set of code points the font maps."""
    return set(TTFont(MONO_PATH).getBestCmap())


def check_coverage(chars):
    """Fail early on any character the font does not cover, rather than
    printing its .notdef on a specimen."""
    missing = sorted({c for c in chars if c != " " and ord(c) not in covered_codepoints()})
    if missing:
        raise ValueError("Not in the font: " + " ".join(f"U+{ord(c):04X}" for c in missing))


# --- The text screen --------------------------------------------------------

# The sixteen colors of the PC's text modes
BLACK = (0x00, 0x00, 0x00)
BLUE = (0x00, 0x00, 0xAA)
GREEN = (0x00, 0xAA, 0x00)
CYAN = (0x00, 0xAA, 0xAA)
RED = (0xAA, 0x00, 0x00)
MAGENTA = (0xAA, 0x00, 0xAA)
BROWN = (0xAA, 0x55, 0x00)
LIGHT_GRAY = (0xAA, 0xAA, 0xAA)
DARK_GRAY = (0x55, 0x55, 0x55)
LIGHT_BLUE = (0x55, 0x55, 0xFF)
LIGHT_GREEN = (0x55, 0xFF, 0x55)
LIGHT_CYAN = (0x55, 0xFF, 0xFF)
LIGHT_RED = (0xFF, 0x55, 0x55)
LIGHT_MAGENTA = (0xFF, 0x55, 0xFF)
YELLOW = (0xFF, 0xFF, 0x55)
WHITE = (0xFF, 0xFF, 0xFF)

BOX_STYLES = {
    "single": "┌─┐│└┘",
    "double": "╔═╗║╚╝",
}


class TextScreen:
    """A text-mode screen: a grid of character cells, each with its own
    character, foreground and background color, and which half of a
    double-height character it shows, if it is one."""

    def __init__(self, cols, rows, fg=LIGHT_GRAY, bg=BLACK):
        self.cols = cols
        self.rows = rows
        self.cells = [[[" ", fg, bg, None] for _ in range(cols)] for _ in range(rows)]

    def put(self, col, row, text, fg=None, bg=None, half=None):
        """Write text from a cell on; a color left out keeps the cell's own."""
        for i, char in enumerate(text):
            if 0 <= col + i < self.cols and 0 <= row < self.rows:
                cell = self.cells[row][col + i]
                cell[0] = char
                cell[1] = fg or cell[1]
                cell[2] = bg or cell[2]
                cell[3] = half

    def put_double(self, col, row, text, fg=None, bg=None):
        """Write double-height text, as Teletext does: the top halves of the
        characters on the row, and their bottom halves on the row below."""
        self.put(col, row, text, fg, bg, "top")
        self.put(col, row + 1, text, fg, bg, "bottom")

    def put_hot(self, col, row, text, fg, hot, bg=None):
        """Write a Turbo Vision label: the letters between tildes are its
        hotkeys, and take the hot color."""
        for i, part in enumerate(text.split("~")):
            self.put(col, row, part, hot if i % 2 else fg, bg)
            col += len(part)

    def fill(self, col, row, width, height, char=" ", fg=None, bg=None):
        """Fill a rectangle of cells with one character."""
        for y in range(row, row + height):
            self.put(col, y, char * width, fg, bg)

    def box(self, col, row, width, height, style="double", fg=None, bg=None):
        """Frame a rectangle with box-drawing characters, and clear inside."""
        tl, h, tr, v, bl, br = BOX_STYLES[style]
        self.put(col, row, tl + h * (width - 2) + tr, fg, bg)
        for y in range(row + 1, row + height - 1):
            self.put(col, y, v, fg, bg)
            self.put(col + 1, y, " " * (width - 2), fg, bg)
            self.put(col + width - 1, y, v, fg, bg)
        self.put(col, row + height - 1, bl + h * (width - 2) + br, fg, bg)

    def shadow(self, col, row, width, height):
        """Cast a Turbo Vision shadow: the cells under it keep their
        characters, dimmed to dark gray on black."""
        for y in range(row, row + height):
            for x in range(col, col + width):
                if 0 <= x < self.cols and 0 <= y < self.rows:
                    self.cells[y][x][1:3] = [DARK_GRAY, BLACK]

    def big_text(self, col, row, text, fg=None, bg=None, scale=1, clear=True):
        """Write text three times over size in sextant characters, or six
        times at scale 2: each cell holds 2 by 3 sextants, each of those
        3 by 3 font pixels of the cell, so the large text keeps the font's
        own pixel proportions. Unless asked to clear, the blank sextant
        cells leave what is under them. Returns the width written, in
        cells."""
        bitmap = big_bitmap(text)
        bitmap = bitmap.resize((bitmap.size[0] * scale, bitmap.size[1] * scale), Image.NEAREST)
        width = (bitmap.size[0] + 1) // 2
        px = bitmap.load()
        for cy in range(bitmap.size[1] // 3):
            for cx in range(width):
                bits = 0
                for i, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]):
                    x, y = cx * 2 + dx, cy * 3 + dy
                    if x < bitmap.size[0] and px[x, y]:
                        bits |= 1 << i
                if bits or clear:
                    self.put(col + cx, row + cy, sextant(bits), fg, bg)

        return width

    def render(self):
        """Render the screen at one image pixel per font pixel."""
        check_coverage({cell[0] for line in self.cells for cell in line})
        img = Image.new("RGB", (self.cols * CELL_WIDTH, self.rows * CELL_HEIGHT))
        for y, line in enumerate(self.cells):
            for x, (char, fg, bg, half) in enumerate(line):
                xy = (x * CELL_WIDTH, y * CELL_HEIGHT)
                img.paste(bg, xy + (xy[0] + CELL_WIDTH, xy[1] + CELL_HEIGHT))
                if char != " ":
                    img.paste(fg, xy, half_cell(char, half))

        return img


def half_cell(char, half):
    """Return a character's cell mask, or one half of it stretched to twice
    its height, every font pixel row doubled."""
    cell = glyph_cell(char)
    if half is None:
        return cell
    tall = cell.resize((CELL_WIDTH, 2 * CELL_HEIGHT), Image.NEAREST)
    top = 0 if half == "top" else CELL_HEIGHT

    return tall.crop((0, top, CELL_WIDTH, top + CELL_HEIGHT))


def big_bitmap(text):
    """Return the font-pixel bitmap of a line of text, its last column, the
    gap after the last character, trimmed off."""
    img = Image.new("L", (len(text) * CELL_WIDTH, CELL_HEIGHT), 0)
    for i, char in enumerate(text):
        img.paste(glyph_cell(char), (i * CELL_WIDTH, 0))

    return img.crop((0, 0, img.size[0] - 1, CELL_HEIGHT))


def sextant(bits):
    """Return the character showing a 2 by 3 block of pixels, bit 0 at the
    top left and bit 5 at the bottom right, row by row. The sextant block
    skips the four patterns that block elements already show."""
    if bits == 0:
        return " "
    if bits == 63:
        return "█"
    if bits == 21:
        return "▌"
    if bits == 42:
        return "▐"

    return chr(0x1FB00 + bits - 1 - (bits > 21) - (bits > 42))


# --- Image helpers ----------------------------------------------------------

CANVAS = (1920, 1080)

# Google Fonts takes its specimen images at this size; they are rendered at
# CANVAS and resampled down
GOOGLE_SIZE = (1600, 900)

JPG_QUALITY = 95

# Every random imperfection is drawn from this seed, so the specimens are
# reproducible
NOISE_SEED = 5

# Lighting fields are evaluated on a small grid and smoothly scaled up
FIELD_SIZE = (64, 36)


def noise_image(size, rng):
    """Build an image of uniform random noise."""
    return Image.frombytes("L", size, rng.randbytes(size[0] * size[1]))


def grain_field(size, rng, depth):
    """Build a fine multiplicative grain field, `depth` levels deep: the
    texture of a phosphor coating or of paper fibers."""
    return noise_image(size, rng).point(lambda v: 255 - depth + v * depth // 255)


def light_field(size, level):
    """Build a smooth full-size lighting field from a function giving the
    level at each point of the field, in normalized coordinates."""
    small = Image.new("L", FIELD_SIZE)
    small.putdata([level(x / (FIELD_SIZE[0] - 1), y / (FIELD_SIZE[1] - 1))
                   for y in range(FIELD_SIZE[1]) for x in range(FIELD_SIZE[0])])

    return small.resize(size, Image.BICUBIC)


def radial_light(size, center_level, edge_level):
    """Build a smooth radial lighting field, brightest in the center."""
    def level(u, v):
        du, dv = u - 0.5, v - 0.5
        d = min(1.0, (du * du * 4 + dv * dv * 4) ** 0.5)

        return round(center_level + (edge_level - center_level) * d ** 1.5)

    return light_field(size, level)


def colorize(mask_img, color):
    """Build an image that fades from black to the color over the mask."""
    return Image.composite(Image.new("RGB", mask_img.size, color),
                           Image.new("RGB", mask_img.size, (0, 0, 0)),
                           mask_img)


def shade(img, field):
    """Darken an image by a grayscale field: falling light, a shadow, a grid."""
    return ImageChops.multiply(img, field.convert("RGB"))


def save_image(img, name, google=True):
    """Save a specimen as tiny5mono-<name>.jpg, along with a -google variant
    resampled to GOOGLE_SIZE where one is called for."""
    stem = "tiny5mono-" + name
    img.save(OUT_PATH / (stem + ".jpg"), optimize=True, quality=JPG_QUALITY)
    if google:
        resampled = img.resize(GOOGLE_SIZE, Image.LANCZOS)
        resampled.save(OUT_PATH / (stem + "-google.jpg"),
                       optimize=True, quality=JPG_QUALITY)


# --- Cathode ray tube -------------------------------------------------------

# A CRT draws a text mode one scanline per font pixel row, and along the
# scanline as fast as its dot clock runs: the pixel's width is a matter of
# timing, not of a grid, and the text modes of the PC ran it fast enough to
# fit 720 pixels across a 4:3 tube, each a little narrower than tall.
#
# Every CRT screen is shown close up, the raster filling the frame but for a
# thin band of the dark tube around it. It is drawn at this many image
# pixels per scanline, and resampled to the canvas once it is finished
CRT_SCANLINE = 8
CRT_MARGIN = 4                  # dark tube around the raster, in scanlines
CRT_GRAIN = 18                  # depth of the phosphor grain, of 255
CRT_BULGE = 0.03                # how far the tube's curve magnifies its center
CRT_REFLECTION = (176, 168, 158)

# Electron beam: a generalized Gaussian across the scanline, whose width
# grows with beam current, so bright lines bloom wider than dim ones
BEAM_EXPONENT = 3.2             # >2 flattens the core and steepens the falloff
BEAM_SIGMA_MIN = 0.32           # as a fraction of the scanline pitch
BEAM_SIGMA_MAX = 0.45
BEAM_FLOOR = 30                 # emission between scanlines, of 255

# A P1 green tube: green at working brightness, the saturated cores washing
# out towards pale green
GREEN_CORE_CURVES = {
    "r": [(0, 0), (0.75, 0.06), (1, 0.62)],
    "g": [(0, 0), (0.25, 0.42), (0.8, 1), (1, 1)],
    "b": [(0, 0), (0.75, 0.10), (1, 0.55)],
}
GREEN_GLOW_CURVES = {
    "r": [(0, 0), (1, 0.08)],
    "g": [(0, 0), (0.3, 0.45), (1, 1)],
    "b": [(0, 0), (1, 0.22)],
}


def curve_lut(points):
    """Build a 256-entry lookup table from piecewise-linear curve points."""
    lut = []
    for value in range(256):
        x = value / 255
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x <= x1:
                t = 0 if x1 == x0 else (x - x0) / (x1 - x0)
                lut.append(round((y0 + (y1 - y0) * t) * 255))
                break

    return lut


def phosphor(intensity, curves):
    """Map beam intensity to phosphor emission through the color curves."""
    return Image.merge("RGB", tuple(intensity.point(curve_lut(curves[c]))
                                    for c in "rgb"))


def beam_profile(size, pitch, sigma):
    """Build the vertical emission profile of the scanning beam: a flat,
    saturated core with a steep falloff into the gap between scanlines."""
    column = Image.new("L", (1, size[1]))
    levels = []
    for y in range(size[1]):
        offset = ((y % pitch) + 0.5) / pitch - 0.5
        weight = math.exp(-abs(offset / sigma) ** BEAM_EXPONENT)
        levels.append(round(BEAM_FLOOR + (255 - BEAM_FLOOR) * weight))
    column.putdata(levels)

    return column.resize(size, Image.NEAREST)


def barrel(img, k, center=(0.5, 0.5)):
    """Warp the image over the bulge of a CRT tube: the center of the bulge,
    given in fractions of the image, is magnified and the raster bows
    outward from it, while the corner furthest from it stays put."""
    width, height = img.size
    columns, rows = 32, 18
    cx, cy = center[0] * width, center[1] * height
    half = (width / 2, height / 2)
    reach = max(((x - cx) / half[0]) ** 2 + ((y - cy) / half[1]) ** 2
                for x in (0, width) for y in (0, height))

    def source(px, py):
        xn = (px - cx) / half[0]
        yn = (py - cy) / half[1]
        f = (1 + k * (xn * xn + yn * yn)) / (1 + k * reach)

        return (cx + xn * f * half[0], cy + yn * f * half[1])

    mesh = []
    for j in range(rows):
        for i in range(columns):
            x0, y0 = width * i // columns, height * j // rows
            x1, y1 = width * (i + 1) // columns, height * (j + 1) // rows
            quad = sum((source(px, py)
                        for px, py in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]), ())
            mesh.append(((x0, y0, x1, y1), quad))

    return img.transform(img.size, Image.MESH, mesh, Image.BILINEAR)


def glass_reflection(size):
    """Build the soft reflection of a bright window across the room."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = size
    draw.rounded_rectangle(xy=[(width * 0.62, height * 0.06),
                               (width * 0.84, height * 0.32)],
                           radius=height // 12, fill=255)
    mask = mask.rotate(6, resample=Image.BILINEAR, center=(width * 0.73, height * 0.19))
    mask = mask.filter(ImageFilter.GaussianBlur(width / 30))

    return mask.point(lambda v: v * 28 // 255)


def tube_size(screen_img, canvas_size, margin=CRT_MARGIN, aspect=PIXEL_ASPECT):
    """Return the size of the tube face the raster is drawn on: the raster
    with its margin of dark tube, widened or deepened to the canvas's
    proportions."""
    width = round(screen_img.size[0] * CRT_SCANLINE * aspect)
    height = screen_img.size[1] * CRT_SCANLINE
    margin = 2 * margin * CRT_SCANLINE
    frame = canvas_size[0] / canvas_size[1]
    tube_height = max(height + margin, round((width + margin) / frame))

    return (round(tube_height * frame), tube_height)


def crt_beam(screen_img, size, aspect=PIXEL_ASPECT, offset=None):
    """Scan a font-pixel image onto a tube face of the given size: one
    scanline per row of font pixels, each font pixel `aspect` as wide as
    the scanline pitch, the raster centered on the face, or with its top
    left at the offset given.

    Returns the beam intensity image, in RGB, before any phosphor.
    """
    pitch = CRT_SCANLINE
    width = round(screen_img.size[0] * pitch * aspect)
    height = screen_img.size[1] * pitch

    # Along the scanline the beam is modulated continuously, so a pixel's
    # edges fall wherever its timing puts them; down the screen, each row
    # is one sweep of the beam
    raster = screen_img.resize((width, screen_img.size[1]), Image.BOX)
    raster = raster.resize((width, height), Image.NEAREST)

    # The spot is round, so it smears horizontally as it sweeps
    raster = raster.filter(ImageFilter.GaussianBlur((pitch * 0.16, pitch * 0.04)))

    # Beam current modulates the scanline width: bright lines run wide,
    # dim ones stay narrow
    lit = raster.convert("L")
    narrow = ImageChops.multiply(raster, beam_profile(raster.size, pitch, BEAM_SIGMA_MIN).convert("RGB"))
    wide = ImageChops.multiply(raster, beam_profile(raster.size, pitch, BEAM_SIGMA_MAX).convert("RGB"))
    beam = Image.composite(wide, narrow, lit)

    face = Image.new("RGB", size)
    face.paste(beam, offset or ((size[0] - width) // 2, (size[1] - height) // 2))

    return face


def finish_tube(img, rng, canvas_size, bulge=CRT_BULGE, bulge_center=(0.5, 0.5),
                reflection=True):
    """The phosphor coating is granular, the tube curves the picture, the
    glass vignettes it, and a window across the room reflects faintly off
    the screen, unless asked not to. The finished face is then resampled to
    the canvas."""
    img = shade(img, grain_field(img.size, rng, CRT_GRAIN))
    img = barrel(img, bulge, bulge_center)
    img = shade(img, radial_light(img.size, 255, 186))
    if reflection:
        img = ImageChops.screen(img, colorize(glass_reflection(img.size), CRT_REFLECTION))

    return img.resize(canvas_size, Image.LANCZOS)


def render_crt_color(screen_img, canvas_size, rng, margin=CRT_MARGIN,
                     aspect=PIXEL_ASPECT, face=None, offset=None, **tube):
    """Render as a color CRT: each gun's beam lights its own phosphor,
    haloed by its own color. The margin of dark tube around the raster is
    given in scanlines, and the font pixel's width in its heights; or the
    face is given outright, with the raster's offset on it, to frame a
    part of a larger raster. Any further settings go to the tube."""
    face = face or tube_size(screen_img, canvas_size, margin, aspect)
    beam = crt_beam(screen_img, face, aspect, offset)

    img = beam
    halo = beam.filter(ImageFilter.GaussianBlur(CRT_SCANLINE * 0.9)).point(lambda v: v * 50 // 100)
    img = ImageChops.screen(img, halo)
    haze = beam.filter(ImageFilter.GaussianBlur(CRT_SCANLINE * 3.0)).point(lambda v: v * 22 // 100)
    img = ImageChops.screen(img, haze)

    return finish_tube(img, rng, canvas_size, **tube)


def render_crt_mono(screen_img, canvas_size, rng, core_curves, glow_curves,
                    margin=CRT_MARGIN, aspect=PIXEL_ASPECT):
    """Render as a monochrome CRT: the beam intensity lights a single
    phosphor through its color curves, cores washing out towards white.
    The margin of dark tube around the raster is given in scanlines, and
    the font pixel's width in its heights."""
    beam = crt_beam(screen_img, tube_size(screen_img, canvas_size, margin, aspect),
                    aspect).convert("L")

    img = phosphor(beam, core_curves)
    halo = beam.filter(ImageFilter.GaussianBlur(CRT_SCANLINE * 1.0)).point(lambda v: v * 70 // 100)
    img = ImageChops.screen(img, phosphor(halo, glow_curves))
    haze = beam.filter(ImageFilter.GaussianBlur(CRT_SCANLINE * 3.2)).point(lambda v: v * 28 // 100)
    img = ImageChops.screen(img, phosphor(haze, glow_curves))

    return finish_tube(img, rng, canvas_size)


# --- VGA text mode: the IDE -------------------------------------------------

# A Turbo Vision IDE in text mode, seen up close, on a screen of the frame's
# own proportions
IDE_GRID = (66, 17)

HERO_CODE = [
    "#include <stdio.h>",
    "#include \"tiny5.h\"",
    "",
    "/* Draw a string on the 6-pixel grid */",
    "void draw_text(int x, int y, const char *s)",
    "{",
    "    while (*s) {",
    "        draw_glyph(x, y, *s++);",
    "        x += GLYPH_WIDTH;",
    "    }",
    "}",
    "",
    "int main(void)",
]

C_KEYWORDS = {"void", "int", "char", "const", "while", "return", "for", "if"}


def put_c(screen, col, row, line, width):
    """Write a line of C with Turbo C's highlighting, clipped to a width."""
    line = line[:width]
    if line.lstrip().startswith("#"):
        screen.put(col, row, line, LIGHT_GREEN)
        return
    if "/*" in line:
        code, comment = line.split("/*", 1)
        put_c(screen, col, row, code, width)
        screen.put(col + len(code), row, "/*" + comment, LIGHT_GRAY)
        return
    x = col
    token = ""
    for char in line + " ":
        if char.isalnum() or char == "_":
            token += char
            continue
        if token:
            screen.put(x, row, token, WHITE if token in C_KEYWORDS else YELLOW)
            x += len(token)
            token = ""
        if x < col + len(line):
            screen.put(x, row, char, LIGHT_CYAN if char == '"' else YELLOW)
            x += 1


def compose_ide():
    """Compose the IDE: the menu bar, the editor window with its source
    file, the About dialog over it, and the status line."""
    cols, rows = IDE_GRID
    s = TextScreen(cols, rows, LIGHT_GRAY, BLUE)

    # The desktop pattern, and the menu bar across the top
    s.fill(0, 1, cols, rows - 2, "░", LIGHT_GRAY, BLUE)
    s.fill(0, 0, cols, 1, " ", BLACK, LIGHT_GRAY)
    x = 1
    for label in [" ≡ ", " ~F~ile ", " ~E~dit ", " ~S~earch ", " ~R~un ",
                  " ~C~ompile ", " ~D~ebug ", " ~T~ools ", " ~O~ptions "]:
        s.put_hot(x, 0, label, BLACK, RED, LIGHT_GRAY)
        x += len(label.replace("~", ""))

    # The editor window, its scroll bars set into the frame
    s.box(0, 1, cols, rows - 2, "double", WHITE, BLUE)
    s.put(2, 1, "[■]", WHITE)
    s.put(3, 1, "■", LIGHT_GREEN)
    s.put(cols - 5, 1, "[↑]", WHITE)
    s.put(cols - 4, 1, "↑", LIGHT_GREEN)
    s.put((cols - 9) // 2, 1, " TINY5.C ", WHITE)
    for i, line in enumerate(HERO_CODE[:rows - 4]):
        put_c(s, 1, 2 + i, line, cols - 2)
    for i, char in enumerate("▲" + "■" + "▒" * (rows - 7) + "▼"):
        s.put(cols - 1, 2 + i, char, CYAN, BLUE)
    s.put(4, rows - 2, " 1:1 ", WHITE)
    s.put(18, rows - 2, "◄■" + "▒" * (cols - 22) + "►", CYAN, BLUE)

    # The About dialog
    width, height = 44, 12
    left, top = (cols - width) // 2, 3
    s.box(left, top, width, height, "double", WHITE, LIGHT_GRAY)
    s.put(left + (width - 7) // 2, top, " About ", BLACK)
    s.put(left + 3, top, "[■]", WHITE)
    s.put(left + 4, top, "■", LIGHT_GREEN)
    title = "Tiny5 Mono"
    title_width = len(title) * CELL_WIDTH // 2
    s.big_text(left + (width - title_width) // 2, top + 2, title, BLUE)
    for i, line in enumerate(["A 5-pixel monospace font", f"Version {VERSION[1:]}"]):
        s.put(left + (width - len(line)) // 2, top + 6 + i, line, BLACK)

    # The OK button, with its half-block shadow
    button = "   OK   "
    bx = left + (width - len(button)) // 2
    by = top + height - 3
    s.put(bx, by, button, WHITE, GREEN)
    s.put(bx + len(button), by, "▄", BLACK)
    s.put(bx + 1, by + 1, "▀" * len(button), BLACK)
    s.shadow(left + width, top + 1, 2, height)
    s.shadow(left + 2, top + height, width, 1)

    # The status line
    s.fill(0, rows - 1, cols, 1, " ", BLACK, LIGHT_GRAY)
    s.put_hot(1, rows - 1, "~F1~ Help  ~F2~ Save  ~F3~ Open  ~Alt-F9~ Compile  ~F9~ Make  ~F10~ Menu",
              BLACK, RED, LIGHT_GRAY)

    return s


def build_ide():
    """The IDE, on a color VGA monitor."""
    rng = random.Random(NOISE_SEED)
    img = render_crt_color(compose_ide().render(), CANVAS, rng)
    save_image(img, "ide")


# --- Green terminal: the message of the day --------------------------------

# The screen is cut to the session, so the session fills the frame, the
# tube showing a little more above and below it than at the sides
HERO_GRID = (58, 14)
HERO_MARGIN = 8                 # dark tube around the raster, in scanlines

GREEN_NORMAL = (180, 180, 180)
GREEN_BRIGHT = (255, 255, 255)
GREEN_DIM = (110, 110, 110)


def compose_green_hero():
    """Compose a session on a green terminal: the message of the day, its
    title in large sextant type, framed in box-drawing characters."""
    cols, rows = HERO_GRID
    s = TextScreen(cols, rows, GREEN_NORMAL, BLACK)
    prompt = "gissio@tiny5:~$ "

    width = 50
    left = (cols - width) // 2
    top = 2
    s.put(left, 0, prompt + "cat /etc/motd", GREEN_NORMAL)
    s.box(left, top, width, 10, "double", GREEN_NORMAL)
    title = "Tiny5 Mono"
    title_width = len(title) * CELL_WIDTH // 2
    s.big_text(left + (width - title_width) // 2, top + 2, title, GREEN_BRIGHT)
    tagline = "A 5-pixel monospace font"
    s.put(left + (width - len(tagline)) // 2, top + 6, tagline, GREEN_NORMAL)
    version = f"version {VERSION[1:]}"
    s.put(left + (width - len(version)) // 2, top + 7, version, GREEN_DIM)
    s.put(left, rows - 1, prompt, GREEN_NORMAL)
    s.put(left + len(prompt), rows - 1, "█", GREEN_BRIGHT)

    return s


def build_green_terminal():
    """The message of the day, on a green-phosphor terminal."""
    rng = random.Random(NOISE_SEED)
    img = render_crt_mono(compose_green_hero().render(), CANVAS, rng,
                          GREEN_CORE_CURVES, GREEN_GLOW_CURVES, HERO_MARGIN)
    save_image(img, "green-terminal")


# --- Norton Commander: the file manager ------------------------------------

# A twin-panel file manager, in the colors of Norton Commander: the font's
# folder on the left, and a quick view of the font on the right
NC_GRID = (70, 18)
NC_FILES = ["..", "DOCS", "SCRIPTS", "SOURCES", "authors  txt", "makefile",
            "ofl      txt", "readme   md", "tiny5bit ttf", "tiny5bld ttf",
            "tiny5crt ttf", "tiny5it  ttf", "tiny5lcd ttf", "tiny5med ttf",
            "tiny5mon bdf", "tiny5mon otf", "tiny5mon ttf", "tiny5mtx ttf"]
NC_SELECTED = "tiny5mon ttf"
NC_KEYS = [("1", "Help"), ("2", "Menu"), ("3", "View"), ("4", "Edit"), ("5", "Copy"),
           ("6", "RenMov"), ("7", "Mkdir"), ("8", "Delete"), ("9", "PullDn"), ("10", "Quit")]


def nc_title(s, left, title, active):
    """Write a panel's title on its top border, highlighted when the panel
    is the active one."""
    x = left + (NC_GRID[0] // 2 - len(title)) // 2
    if active:
        s.put(x, 0, title, BLACK, CYAN)
    else:
        s.put(x, 0, title, LIGHT_CYAN)


def compose_commander():
    """Compose the file manager: the file panel, the quick view panel, the
    command line and the function key bar."""
    cols, rows = NC_GRID
    s = TextScreen(cols, rows, LIGHT_CYAN, BLUE)
    width = cols // 2

    # The file panel, in two columns of names, filled column by column, and
    # its title written over the join of the columns on the border
    height = rows - 2
    s.box(0, 0, width, height, "double", LIGHT_CYAN, BLUE)
    split = width // 2
    s.put(split, 0, "╤", LIGHT_CYAN)
    for row in range(1, height - 3):
        s.put(split, row, "│", LIGHT_CYAN)
    s.put(0, height - 3, "╟" + "─" * (width - 2) + "╢", LIGHT_CYAN)
    s.put(split, height - 3, "┴", LIGHT_CYAN)
    for x0, x1 in [(1, split), (split + 1, width - 1)]:
        s.put(x0 + (x1 - x0 - 4) // 2, 1, "Name", YELLOW)
    per_column = height - 5
    for i, name in enumerate(NC_FILES):
        column, row = divmod(i, per_column)
        x0, x1 = [(1, split), (split + 1, width - 1)][column]
        color = WHITE if name.isupper() or name == ".." else LIGHT_CYAN
        if name == NC_SELECTED:
            s.put(x0, 2 + row, " " * (x1 - x0), BLACK, CYAN)
            s.put(x0 + 2, 2 + row, name, BLACK, CYAN)
        else:
            s.put(x0 + 2, 2 + row, name, color)
    s.put(1, height - 2, " tiny5mon ttf  573944 7-10-26", LIGHT_CYAN)
    nc_title(s, 0, " C:\\TINY5 ", True)

    # The quick view of the font
    s.box(width, 0, width, height, "double", LIGHT_CYAN, BLUE)
    nc_title(s, width, " Quick View ", False)
    for row, word in [(2, "Tiny5"), (6, "Mono")]:
        word_width = (big_bitmap(word).size[0] + 1) // 2
        s.big_text(width + 1 + (width - 2 - word_width) // 2, row, word, YELLOW)
    for row, line, color in [(10, "A 5-pixel monospace font", LIGHT_CYAN),
                             (12, "2287 glyphs  974 languages", WHITE),
                             (14, "wght wdth ital ROND BLED JITT", CYAN)]:
        s.put(width + 1 + (width - 2 - len(line)) // 2, row, line, color)

    # The command line, and the function keys
    s.fill(0, rows - 2, cols, 2, " ", LIGHT_GRAY, BLACK)
    s.put(0, rows - 2, "C:\\TINY5>_", LIGHT_GRAY)
    for i, (key, label) in enumerate(NC_KEYS):
        x = i * 7
        s.put(x, rows - 1, key, LIGHT_GRAY, BLACK)
        s.put(x + len(key), rows - 1, label.ljust(7 - len(key)), BLACK, CYAN)

    return s


def build_hero_commander():
    """The file manager prototype, on a color VGA monitor."""
    rng = random.Random(NOISE_SEED)
    save_image(render_crt_color(compose_commander().render(), CANVAS, rng),
               "hero-commander")


# --- Modern editor on a color CRT: the code ---------------------------------

# A modern code editor in its dark theme, seen up close on a color tube: the
# tab bar, the breadcrumbs, the source with its line numbers, and the status
# bar
ED_BG = (21, 21, 21)
ED_TABS = (37, 37, 38)
ED_TAB_IDLE = (45, 45, 45)
ED_LINE = (40, 42, 44)          # the cursor's line
ED_GUTTER = (110, 110, 110)
ED_GUTTER_ACTIVE = (198, 198, 198)
ED_STATUS = (0, 122, 204)
ED_TEXT = (212, 212, 212)
ED_MUTED = (150, 150, 150)
ED_KEYWORD = (86, 156, 214)
ED_CONTROL = (197, 134, 192)
ED_FUNCTION = (220, 220, 170)
ED_STRING = (206, 145, 120)
ED_NUMBER = (181, 206, 168)
ED_COMMENT = (106, 153, 85)
ED_TITLE = (130, 196, 98)       # the banner's large type, a brighter comment

ED_SOURCE = [
    "/*",
    " *",
    " *",
    " *",
    " *  A 5-pixel monospace font",
    " */",
    "#include <stdio.h>",
    "",
    "int main(void) {",
    "    puts(\"Hello, world!\");  // 5 pixels tall",
    "    return 0;",
    "}",
]
ED_CURSOR_LINE = 10             # 0-based: the line being typed
ED_KEYWORDS = {"int", "void", "char", "const", "unsigned", "static"}
ED_CONTROLS = {"return", "if", "else", "for", "while"}
ED_GAIN = 1.35                  # the tube is driven a little hot, as CRTs
                                # were, to stand up to a lit room
ED_GRID = (54, 14)              # wide enough to fill the frame, once the
                                # rows fill its height
ED_CODE_COL = 8                 # the source's first column, after the gutter
ED_GUTTER_GAP = 3               # columns between a line number and its line
ED_CROP = 0                     # scanlines cut off the top and the foot,
                                # besides what the tube's curve takes
ED_BULGE = 0.02                 # a gentle curve, its center right of the
ED_BULGE_CENTER = (0.62, 0.5)   # middle, as the camera stands to the right


def put_code(s, col, row, line, bg):
    """Write a line of C in the dark theme's colors."""
    stripped = line.lstrip()
    if stripped.startswith(("/*", "*")):
        s.put(col, row, line, ED_COMMENT, bg)
        return
    if stripped.startswith("#"):
        directive, _, rest = line.partition(" ")
        s.put(col, row, directive, ED_CONTROL, bg)
        s.put(col + len(directive) + 1, row, rest, ED_STRING, bg)
        return
    code, slashes, comment = line.partition("//")
    x = col
    tokens = re.findall(r'"[^"]*"|\w+|\W', code)
    for i, token in enumerate(tokens):
        following = next((t for t in tokens[i + 1:] if t != " "), "")
        if token.startswith('"'):
            color = ED_STRING
        elif token in ED_KEYWORDS:
            color = ED_KEYWORD
        elif token in ED_CONTROLS:
            color = ED_CONTROL
        elif token.isdigit():
            color = ED_NUMBER
        elif token[0].isalpha() and following == "(":
            color = ED_FUNCTION
        else:
            color = ED_TEXT
        s.put(x, row, token, color, bg)
        x += len(token)
    if slashes:
        s.put(x, row, slashes + comment, ED_COMMENT, bg)


def compose_editor_hero():
    """Compose the editor."""
    cols, rows = ED_GRID
    s = TextScreen(cols, rows, ED_TEXT, ED_BG)

    # The tab bar: the file being edited, and one more behind it
    s.fill(0, 0, cols, 1, " ", ED_MUTED, ED_TABS)
    s.put(0, 0, " tiny5.c ● ", ED_TEXT, ED_BG)
    s.put(11, 0, " README.md ", ED_MUTED, ED_TAB_IDLE)

    # The breadcrumbs
    s.put(ED_CODE_COL, 1, "src > tiny5.c > main", ED_MUTED)

    # The source, its line numbers in a wide gutter, the cursor's line lit
    top = 2
    height = rows - top - 1
    for i, line in enumerate(ED_SOURCE[:height]):
        bg = ED_LINE if i == ED_CURSOR_LINE else ED_BG
        if bg != ED_BG:
            s.fill(0, top + i, cols, 1, " ", ED_TEXT, bg)
        number = f"{i + 1:{ED_CODE_COL - ED_GUTTER_GAP}}"
        s.put(0, top + i, number, ED_GUTTER_ACTIVE if i == ED_CURSOR_LINE else ED_GUTTER, bg)
        put_code(s, ED_CODE_COL, top + i, line, bg)
    s.big_text(ED_CODE_COL + 4, top + 1, "Tiny5 Mono", ED_TITLE)
    cursor = ED_CODE_COL + len(ED_SOURCE[ED_CURSOR_LINE])
    s.put(cursor, top + ED_CURSOR_LINE, "▏", ED_TEXT, ED_LINE)

    # The status bar
    s.fill(0, rows - 1, cols, 1, " ", WHITE, ED_STATUS)
    s.put(2, rows - 1, "main", WHITE, ED_STATUS)
    right = f"Ln {ED_CURSOR_LINE + 1}, Col {len(ED_SOURCE[ED_CURSOR_LINE]) + 1}   UTF-8   LF   C  "
    s.put(cols - len(right), rows - 1, right, WHITE, ED_STATUS)

    return s


def build_editor():
    """The modern editor, on a color CRT, seen from close by: the tab
    bar against the top of the frame and the status bar against its foot,
    the frame taking in as much of the width as that leaves, and the
    tube's curve taking a little off every edge."""
    save_image(render_closeup(compose_editor_hero()), "sample1")


def render_closeup(screen):
    """Render a screen on the color CRT as the hero shows it, from close by:
    driven a little hot, its first row against the top of the frame and its
    last against the foot, the frame taking in as much of the width as that
    leaves, and the tube's gentle curve taking a little off every edge.

    The screen is a text screen, or a raster already drawn, one image pixel
    per raster pixel."""
    rng = random.Random(NOISE_SEED)
    img = screen.render() if isinstance(screen, TextScreen) else screen
    img = img.point(lambda v: min(255, round(v * ED_GAIN)))
    width = round(img.size[0] * CRT_SCANLINE * PIXEL_ASPECT)
    height = img.size[1] * CRT_SCANLINE
    face_height = height - 2 * ED_CROP * CRT_SCANLINE
    face = (round(face_height * CANVAS[0] / CANVAS[1]), face_height)
    offset = ((face[0] - width) // 2, -ED_CROP * CRT_SCANLINE)

    return render_crt_color(img, CANVAS, rng, face=face, offset=offset,
                            bulge=ED_BULGE, bulge_center=ED_BULGE_CENTER,
                            reflection=False)


# --- System monitor: the terminal ------------------------------------------

# A system monitor running in a terminal, on the editor's tube and at its
# zoom: the processor's load as a filled graph in sextants and as a meter
# of squares per core, the memory in meters, and the processes in a list,
# each in its rounded box
TUI_CPU_BOX = (86, 156, 214)
TUI_MEM_BOX = (206, 145, 120)
TUI_PROC_BOX = (197, 134, 192)
TUI_LOW = (78, 201, 176)        # the load gradient, from idle to flat out
TUI_MID = (220, 220, 120)
TUI_HIGH = (244, 96, 86)
TUI_SELECTED = (38, 79, 120)
TUI_METER_OFF = (52, 52, 54)    # a meter's squares beyond its level

TUI_CORES = [0.52, 0.86, 0.31, 0.67]
TUI_MEMORY = [("Used", 0.61, "9.8G"), ("Avail", 0.39, "6.2G"),
              ("Cache", 0.21, "3.4G"), ("Swap", 0.02, "0.3G")]
# A developer's machine at work, the busiest first; the ids are the first
# digits of pi, e, the golden ratio and the square root of 2
TUI_PROCESSES = [(3141, "cargo", 41.7), (2718, "node", 18.3), (1618, "python3", 9.6),
                 (1414, "nvim", 2.1), (1, "systemd", 0.3)]
TUI_SELECTED_ROW = 0
TUI_MARGIN = 1                  # blank columns either side, so the tube's
                                # curve keeps the boxes' sides in the frame


def rounded_box(s, col, row, width, height, title, color, bg):
    """Draw a box with rounded corners, its title set into the top edge."""
    s.put(col, row, "╭" + "─" * (width - 2) + "╮", color, bg)
    for y in range(row + 1, row + height - 1):
        s.put(col, y, "│", color, bg)
        s.put(col + width - 1, y, "│", color, bg)
    s.put(col, row + height - 1, "╰" + "─" * (width - 2) + "╯", color, bg)
    s.put(col + 2, row, "┤", color, bg)
    s.put(col + 3, row, title, ED_TEXT, bg)
    s.put(col + 3 + len(title), row, "├", color, bg)


def load_color(level):
    """Return the gradient's color at a load level, from 0 to 1."""
    if level < 0.5:
        a, b, t = TUI_LOW, TUI_MID, level * 2
    else:
        a, b, t = TUI_MID, TUI_HIGH, level * 2 - 1

    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def meter(s, col, row, width, level, bg):
    """Draw a meter as a row of squares: lit along the gradient as far as
    the level reaches, and dim beyond."""
    lit = round(level * width)
    for i in range(width):
        color = load_color(i / (width - 1)) if i < lit else TUI_METER_OFF
        s.put(col + i, row, "■", color, bg)


def load_history(count, rng):
    """Return a processor's recent load: slow swells, with a little noise."""
    return [max(0.04, min(1.0, 0.45 + 0.25 * math.sin(i / 5.0) + 0.18 * math.sin(i / 1.7 + 1)
                          + rng.uniform(-0.08, 0.08)))
            for i in range(count)]


def load_graph(s, col, row, width, height, history, bg):
    """Draw the load as a filled graph in sextants: two samples a cell
    across, and three steps a cell up, each row of cells in the gradient's
    color for its height."""
    steps = height * 3
    for cy in range(height):
        color = load_color(1 - (cy + 0.5) / height)
        for cx in range(width):
            bits = 0
            for dx in range(2):
                level = round(history[cx * 2 + dx] * steps)
                for dy in range(3):
                    if steps - (cy * 3 + dy) <= level:
                        bits |= 1 << (dy * 2 + dx)
            s.put(col + cx, row + cy, sextant(bits), color, bg)


def compose_monitor():
    """Compose the monitor: the processor box across the top, the memory
    and the process boxes side by side below it, inset from the screen's
    sides."""
    cols, rows = ED_GRID[0] - 2 * TUI_MARGIN, ED_GRID[1]
    bg = ED_BG
    s = TextScreen(cols, rows, ED_TEXT, bg)
    rng = random.Random(NOISE_SEED)

    # The processor: the graph, and a meter per core
    rounded_box(s, 0, 0, cols, 6, "cpu", TUI_CPU_BOX, bg)
    s.put(cols - 12, 0, "┤", TUI_CPU_BOX, bg)
    s.put(cols - 11, 0, "3.20 GHz", ED_MUTED, bg)
    s.put(cols - 3, 0, "├", TUI_CPU_BOX, bg)
    graph_width = cols - 23
    load_graph(s, 2, 1, graph_width, 4, load_history(graph_width * 2, rng), bg)
    for i, level in enumerate(TUI_CORES):
        x = cols - 19
        s.put(x, 1 + i, f"C{i}", ED_MUTED, bg)
        meter(s, x + 3, 1 + i, 10, level, bg)
        s.put(x + 14, 1 + i, f"{round(level * 100):3}%", ED_TEXT, bg)

    # The memory, in meters
    half = cols // 2
    rounded_box(s, 0, 6, half, rows - 6, "mem", TUI_MEM_BOX, bg)
    s.put(2, 7, "Total", ED_MUTED, bg)
    s.put(half - 7, 7, "16.0G", ED_TEXT, bg)
    for i, (name, level, amount) in enumerate(TUI_MEMORY):
        s.put(2, 9 + i, name, ED_MUTED, bg)
        meter(s, 8, 9 + i, half - 16, level, bg)
        s.put(half - 6, 9 + i, amount, ED_TEXT, bg)

    # The processes, the selected one lit
    width = cols - half
    rounded_box(s, half, 6, width, rows - 6, "proc", TUI_PROC_BOX, bg)
    s.put(half + 2, 7, "  pid program     cpu%", ED_MUTED, bg)
    for i, (pid, name, load) in enumerate(TUI_PROCESSES):
        row_bg = TUI_SELECTED if i == TUI_SELECTED_ROW else bg
        if row_bg != bg:
            s.fill(half + 1, 8 + i, width - 2, 1, " ", ED_TEXT, row_bg)
        s.put(half + 2, 8 + i, f"{pid:5} {name:<11}{load:5.1f}",
              WHITE if i == TUI_SELECTED_ROW else ED_TEXT, row_bg)

    # Key hints, set into the foot of the boxes
    for x, hint, color in [(2, "q quit", TUI_MEM_BOX), (half + 2, "↑↓ select", TUI_PROC_BOX)]:
        s.put(x, rows - 1, "┤", color, bg)
        s.put(x + 1, rows - 1, hint, ED_MUTED, bg)
        s.put(x + 1 + len(hint), rows - 1, "├", color, bg)

    framed = TextScreen(ED_GRID[0], rows, ED_TEXT, bg)
    for y, line in enumerate(s.cells):
        framed.cells[y][TUI_MARGIN:TUI_MARGIN + cols] = line

    return framed


def build_monitor():
    """The system monitor, on the editor's tube."""
    save_image(render_closeup(compose_monitor()), "sample4")


# --- Character LCD: the character set ---------------------------------------

# A character LCD module, 20 characters by 4, seen up close: each character
# a block of 5 by 8 dots, a dot's width apart from the next, and the dots
# taller than wide. Tiny5 Mono's cell is that block with its gap: the glyphs
# fill the 5 by 8 dots, and the module draws nothing in the gap, nor in the
# cell's bottom row
LCD_DOTS = (5, 8)
LCD_PIXEL = 20                  # image pixels per dot pitch, down
LCD_SUPER = 3                   # the module is drawn this many times over
                                # size, and reduced once it is finished
LCD_DOT_FILL = 0.88             # how much of its pitch a dot covers
LCD_MARGIN = (4, 4)             # glass around the dots, in dot pitches
LCD_BG = (178, 204, 40)         # yellow-green STN, lit from behind
LCD_INK = (34, 50, 12)
LCD_SEAL = 1.4                  # glass past the viewing area, beyond the
                                # backlight's reach, in dot pitches
LCD_FRAME = (24, 24, 26)        # the black metal frame folded over the glass
LCD_FRAME_LIP = 0.35            # its folded edge, in dot pitches
LCD_FRAME_GRAIN = 10            # depth of the frame's paint texture, of 255

# The character set: the Latin alphabet, each capital over its lowercase,
# then a Greek and a Cyrillic pair, and the digits over their shifted
# symbols, as on a keyboard's top row
LCD_CHARSET = [
    "ABCDEFGHIJKLMNOPQRST",
    "abcdefghijklmnopqrst",
    "UVWXYZ ΩЖ 1234567890",
    "uvwxyz ωж !@#$%^&*()",
]


def sheen(size, strength=10):
    """Build a soft diagonal highlight, light catching the polarizer film."""
    return light_field(size, lambda u, v: round(
        strength * math.exp(-(u + v - 0.55) ** 2 / 0.065)))


def render_lcd(lines, canvas_size):
    """Render as a backlit character LCD in its bezel: the dots darken the
    liquid crystal over the backlight, casting a soft shadow on the
    reflector behind, and the unlit dots of every character show faintly."""
    cols, rows = max(len(line) for line in lines), len(lines)
    check_coverage({char for line in lines for char in line})
    pitch_y = LCD_PIXEL * LCD_SUPER
    pitch_x = pitch_y * PIXEL_ASPECT
    margin_x, margin_y = LCD_MARGIN[0] * pitch_x, LCD_MARGIN[1] * pitch_y

    # The glass shows the dots and its margin, but not the gap after the
    # last character, nor the row of the cell below the dots
    width = round(cols * CELL_WIDTH * pitch_x - pitch_x + 2 * margin_x)
    height = round(rows * CELL_HEIGHT * pitch_y - pitch_y + 2 * margin_y)

    ghost = Image.new("L", (width, height), 0)
    lit = Image.new("L", (width, height), 0)
    ghost_draw, lit_draw = ImageDraw.Draw(ghost), ImageDraw.Draw(lit)
    inset_x = (1 - LCD_DOT_FILL) / 2 * pitch_x
    inset_y = (1 - LCD_DOT_FILL) / 2 * pitch_y
    for row, line in enumerate(lines):
        for col, char in enumerate(line.ljust(cols)):
            cell = glyph_cell(char).load()
            for dy in range(LCD_DOTS[1]):
                for dx in range(LCD_DOTS[0]):
                    x = margin_x + (col * CELL_WIDTH + dx) * pitch_x
                    y = margin_y + (row * CELL_HEIGHT + dy) * pitch_y
                    xy = [x + inset_x, y + inset_y,
                          x + pitch_x - inset_x, y + pitch_y - inset_y]
                    ghost_draw.rounded_rectangle(xy, radius=pitch_x * 0.12, fill=255)
                    if cell[dx, dy]:
                        lit_draw.rounded_rectangle(xy, radius=pitch_x * 0.12, fill=255)
    lit = lit.filter(ImageFilter.GaussianBlur(LCD_SUPER * 0.6))

    glass = Image.new("RGB", (width, height), LCD_BG)
    glass = shade(glass, radial_light(glass.size, 255, 200))
    glass = shade(glass, ghost.point(lambda v: 255 - v * 14 // 255))
    shadow = lit.filter(ImageFilter.GaussianBlur(pitch_y * 0.35))
    shadow = ImageChops.offset(shadow, round(pitch_x * 0.3), round(pitch_y * 0.4))
    glass = shade(glass, shadow.point(lambda v: 255 - v * 70 // 255))
    glass.paste(Image.new("RGB", glass.size, LCD_INK), (0, 0),
                lit.point(lambda v: v * 240 // 255))
    # Past the viewing area the glass runs on under the frame, out of the
    # backlight's reach: the same polarizer, but unlit, and the frame's
    # edge shades it
    seal = round(LCD_SEAL * pitch_y)
    full = (canvas_size[0] * LCD_SUPER, canvas_size[1] * LCD_SUPER)
    left, top = (full[0] - width) // 2, (full[1] - height) // 2
    window = [(left - seal, top - seal), (left + width + seal, top + height + seal)]
    img = Image.new("RGB", full, tuple(c * 3 // 5 for c in LCD_BG))
    img.paste(glass, (left, top))
    falloff = Image.new("L", full, 0)
    ImageDraw.Draw(falloff).rectangle(xy=[(left, top), (left + width, top + height)], fill=255)
    falloff = falloff.filter(ImageFilter.GaussianBlur(seal * 0.45))
    img = Image.composite(img, shade(img, Image.new("L", full, 150)), falloff)

    # The frame: black paint over metal, with a fine texture, lit from
    # above; its folded edge catches the light along the bottom of the
    # window, and throws a shadow onto the glass along the top
    frame = Image.new("RGB", full, LCD_FRAME)
    frame = shade(frame, light_field(full, lambda u, v: round(255 - 60 * v)))
    frame = shade(frame, grain_field(full, random.Random(NOISE_SEED), LCD_FRAME_GRAIN))
    lip = round(LCD_FRAME_LIP * pitch_y)
    draw = ImageDraw.Draw(frame)
    (x0, y0), (x1, y1) = window
    draw.rectangle(xy=[(x0 - lip, y1), (x1 + lip, y1 + lip)], fill=(70, 70, 74))
    draw.rectangle(xy=[(x1, y0 - lip), (x1 + lip, y1 + lip)], fill=(46, 46, 50))
    draw.rectangle(xy=[(x0 - lip, y0 - lip), (x1 + lip, y0)], fill=(10, 10, 11))
    draw.rectangle(xy=[(x0 - lip, y0 - lip), (x0, y1 + lip)], fill=(14, 14, 15))
    cutout = Image.new("L", full, 255)
    ImageDraw.Draw(cutout).rectangle(xy=window, fill=0)
    shadow = cutout.filter(ImageFilter.GaussianBlur(lip))
    shadow = ImageChops.offset(shadow, 0, lip)
    img = shade(img, shadow.point(lambda v: 255 - v * 90 // 255))
    img = Image.composite(frame, img, cutout)

    return img.reduce(LCD_SUPER)


def build_charset():
    """The character set, on a 20 by 4 character LCD."""
    save_image(render_lcd(LCD_CHARSET, CANVAS), "sample2")


# --- Thermal printer: the size ramp -----------------------------------------

# A receipt off an 80 mm thermal printer, seen up close. Its head is a row of
# 576 heating elements, and a font pixel takes 2 of them across and 3 lines
# of the feed down at 6 pt, as near as whole dots come to its 89:128 shape;
# at 6 pt the receipt sets the printer's own 48 columns. The same test
# string runs from headline down to that size, broken on whole words, each
# size labeled in points and in pixels
RAMP_SCALES = [5, 3, 2, 1]
RAMP_LINES = [1, 1, 2, 2]      # lines of the test string at each size
RAMP_TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

THERMAL_DOTS = 576              # heating elements across the head
THERMAL_PIXEL = (2, 3)          # dots per font pixel at 6 pt, across and down
THERMAL_LENGTH = 540            # dots of the feed the printing takes
THERMAL_FEED = (56, 72)         # blank paper fed before and after it, in dots
THERMAL_MARGIN = 32             # paper past the printing each side, in dots
THERMAL_TEXTURE = 4             # the flat receipt's pixels per dot
THERMAL_PAPER = (246, 246, 241) # coated thermal paper, a cool white
THERMAL_INK = (30, 30, 36)
THERMAL_DOT_SHAPE = (1.25, 0.95)    # a dot's mark, in dot pitches across and down
THERMAL_WEAR = 0.08             # how much fainter the far end of the head burns
THERMAL_WEAK_DOT = 0.63         # where along the head an element is failing
THERMAL_LABEL_LEAD = 8          # dots from a size's label down to its text
THERMAL_TOOTH = 12              # the tear bar's tooth pitch, in dots
THERMAL_TOOTH_DEPTH = 1.6       # how deep its teeth tear, in dots

# The receipt on the table, in dots: its ends curl up off the table over
# THERMAL_CURL_LENGTH, to the angles given, in degrees, at the top and foot;
# the rest of it waves a little, and its long edges lift a hair
THERMAL_CURL_LENGTH = 170
THERMAL_CURL = (55, 65)
THERMAL_LIFT = 0.8              # the middle's height above the table
THERMAL_WAVE = 3.0              # the waves' depth
THERMAL_EDGE_LIFT = 2.5
THERMAL_TILT = -3               # the receipt's turn on the table, in degrees

# The scene is path traced with Mitsuba, in millimeters, 8 dots to the
# millimeter: a table of black-stained ash, a lamp to the left of the
# receipt, the dim room around, and a camera all but straight above
DOTS_PER_MM = 8
SCENE_TABLE = (14, 14, 15)                      # the black stain
SCENE_FINISH = 0.3                              # the satin finish's roughness
SCENE_TABLE_SIZE = (400, 267)                   # in millimeters
SCENE_TABLE_TEXTURE = 4096                      # pixels across
SCENE_RING = 10                                 # a ring's width, in millimeters
SCENE_GRAIN_WAVE = 30                           # how far the rings wander, in mm
SCENE_PORE_LENGTH = 12                          # a pore streak's length, in mm
SCENE_PORE_WIDTH = 0.5                          # and its width
SCENE_PORE_DEPTH = 0.5                          # its depth below the finish, in dots
SCENE_RELIEF_BLUR = 0.1                         # the finish softens it, in mm
SCENE_RING_DEPTH = 0.1                         # the early wood's, in dots
# The lamp hangs close and small, so its light pools on the receipt and
# falls away across the table; it stands low to the left, where the
# varnish's reflection of it turns away from the camera
SCENE_LAMP = (-213, 57, 172)                    # its center, in millimeters
SCENE_LAMP_SIZE = 70                            # its side, in millimeters
SCENE_LAMP_RADIANCE = (65, 60, 54)
SCENE_ROOM_RADIANCE = (0.025, 0.027, 0.03)
SCENE_CAMERA_HEIGHT = 450       # millimeters above the table
SCENE_CAMERA_TILT = 8           # its tilt from straight down, in degrees
SCENE_VIEW = 164                # the width of table in view, in millimeters
SCENE_APERTURE = 4              # the lens's aperture radius, in millimeters
SCENE_SAMPLES = 256             # paths per pixel
SCENE_VIGNETTE = 0.3            # how much light the lens loses at the corners


def thermal_width(text, scale):
    """Return the width of a line's ink, in dots, the gap after its last
    character left off."""
    return (len(text) * CELL_WIDTH - 1) * THERMAL_PIXEL[0] * scale


def thermal_text(grid, x, cap_line, scale, text):
    """Print a line into the dot grid, x its left edge and cap_line the top
    of its capitals, in dots."""
    across, down = THERMAL_PIXEL[0] * scale, THERMAL_PIXEL[1] * scale
    top = cap_line - (CELL_ASCENT - CAP_PIXELS) * down
    for i, char in enumerate(text):
        if char != " ":
            cell = glyph_cell(char).resize((CELL_WIDTH * across, CELL_HEIGHT * down), Image.NEAREST)
            grid.paste(255, (x + i * CELL_WIDTH * across, top), cell)


def wrap_words(text, columns):
    """Break text into lines of at most `columns` characters, on spaces."""
    lines = [""]
    for word in text.split():
        if lines[-1] and len(lines[-1]) + 1 + len(word) > columns:
            lines.append(word)
        else:
            lines[-1] = (lines[-1] + " " + word).strip()

    return lines


def compose_receipt(rows):
    """Compose the receipt on the head's dot grid, `rows` dots long: a
    centered header, a rule, and the test string at each size under its
    label, then a rule.

    The lines are spaced on the ink, from cap line to descender line. A
    size's lines keep the font's own line spacing, its label stands a fixed
    lead above it, and the space left over spreads evenly between the
    groups, and above the first and below the last."""
    check_coverage(set(RAMP_TEXT))

    def columns(scale):
        return THERMAL_DOTS // (CELL_WIDTH * THERMAL_PIXEL[0] * scale)

    # Each line: its scale, its text, centered or not, the text set flush
    # right on it if any, and its lead above, None where it shares the
    # space left over
    rule = (1, "-" * columns(1), False, None, None)
    lines = [(2, "Tiny5 Mono", True, None, None),
             (1, "Size test " + VERSION, True, None, THERMAL_LABEL_LEAD),
             rule]
    for scale, count in zip(RAMP_SCALES, RAMP_LINES):
        lines.append((1, f"{scale * 6} pt", False, f"{scale * 8} px", None))
        leading = (CELL_HEIGHT - CAP_PIXELS - 1) * THERMAL_PIXEL[1] * scale
        for i, text in enumerate(wrap_words(RAMP_TEXT, columns(scale))[:count]):
            lines.append((scale, text, False, None, leading if i else THERMAL_LABEL_LEAD))
    lines.append(rule)

    def ink(scale):
        return (CAP_PIXELS + 1) * THERMAL_PIXEL[1] * scale

    shared = [lead for *_, lead in lines].count(None) + 1
    filled = sum(ink(scale) + (lead or 0) for scale, *_, lead in lines)
    gap = (rows - filled) // shared

    grid = Image.new("L", (THERMAL_DOTS, rows), 0)
    y = 0
    for scale, text, centered, right, lead in lines:
        y += gap if lead is None else lead
        x = (THERMAL_DOTS - thermal_width(text, scale)) // 2 if centered else 0
        thermal_text(grid, x, y, scale, text)
        if right:
            thermal_text(grid, THERMAL_DOTS - thermal_width(right, scale), y, scale, right)
        y += ink(scale)

    return grid


def print_receipt(grid, rng):
    """Print the dot grid onto a flat strip of thermal paper, and return it
    as an RGBA texture, THERMAL_TEXTURE pixels to the dot, torn off the roll
    at both ends.

    Each dot burns as dark as its element and its line allow: the elements
    differ a little, the head wears toward its far end, and one element is
    failing, leaving a faint streak down the print; the feed's energy drifts
    from line to line. Each dot burns a rounded mark, a little short of the
    next line of the feed, and the coating mottles the black."""
    size = grid.size
    pitch = THERMAL_TEXTURE

    elements = [rng.uniform(0.93, 1.0) * (1 - THERMAL_WEAR * c / size[0])
                for c in range(size[0])]
    weak = round(THERMAL_WEAK_DOT * size[0])
    elements[weak] *= 0.4
    elements[weak + 1] *= 0.8
    head = Image.new("L", (size[0], 1))
    head.putdata([round(255 * v) for v in elements])
    feed = noise_image((1, size[1] // 12 + 2), rng).point(lambda v: 232 + v * 23 // 255)
    feed = feed.resize((1, size[1]), Image.BICUBIC, box=(0, 0.5, 1, size[1] / 12 + 0.5))
    dots = ImageChops.multiply(grid, head.resize(size, Image.NEAREST))
    dots = ImageChops.multiply(dots, feed.resize(size, Image.NEAREST))
    dots = ImageChops.multiply(dots, noise_image(size, rng).point(lambda v: 225 + v * 30 // 255))

    ink = Image.new("L", (size[0] * pitch, size[1] * pitch), 0)
    draw = ImageDraw.Draw(ink)
    level = dots.load()
    rx, ry = pitch * THERMAL_DOT_SHAPE[0] / 2, pitch * THERMAL_DOT_SHAPE[1] / 2
    for y in range(size[1]):
        for x in range(size[0]):
            if level[x, y]:
                cx = (x + 0.5) * pitch + rng.uniform(-0.08, 0.08) * pitch
                cy = (y + 0.5) * pitch
                draw.ellipse(xy=[cx - rx, cy - ry, cx + rx, cy + ry], fill=level[x, y])
    ink = ink.filter(ImageFilter.GaussianBlur(pitch * 0.12))
    ink = ink.point(lambda v: min(255, max(0, (v - 60) * 8 // 5)))
    mottle = noise_image(ink.size, rng).filter(ImageFilter.GaussianBlur(pitch * 0.4))
    mottle = mottle.point(lambda v: min(255, max(0, 255 - (128 - v) * 3)))
    ink = ImageChops.multiply(ink, mottle.point(lambda v: 205 + v * 50 // 255))

    # The paper: the print centered across it, with the feed before and
    # after it
    margin, feed_top = THERMAL_MARGIN * pitch, THERMAL_FEED[0] * pitch
    paper_size = (ink.width + 2 * margin, ink.height + feed_top + THERMAL_FEED[1] * pitch)
    paper = Image.new("RGB", paper_size, THERMAL_PAPER)
    paper = shade(paper, grain_field(paper_size, rng, 5))
    paper.paste(Image.new("RGB", ink.size, THERMAL_INK), (margin, feed_top), ink)

    # Torn across the tear bar's teeth at both ends: a fine zigzag, each
    # tooth a little different
    alpha = Image.new("L", paper_size, 255)
    draw = ImageDraw.Draw(alpha)
    tooth = THERMAL_TOOTH * pitch
    for edge in (0, paper_size[1]):
        inward = 1 if edge == 0 else -1
        points = [(0, edge - inward)]
        for i in range(round(paper_size[0] / tooth * 2) + 1):
            depth = (THERMAL_TOOTH_DEPTH if i % 2 else 0.2) * rng.uniform(0.6, 1.4)
            points.append((i * tooth / 2, edge + inward * depth * pitch))
        points.append((paper_size[0], edge - inward))
        draw.polygon(points, fill=0)
    paper.putalpha(alpha)

    return paper


def receipt_profile(length):
    """Return the receipt's profile along its length, sampled every dot of
    the paper: how far down the table each dot lies, how high above it, and
    the angle the paper makes with the table there.

    The paper remembers the roll, and both ends curl up off the table, the
    curl tightening toward the edge; the middle rests on the table, all but
    a hair above it, as paper never lies quite flat."""
    s = np.arange(length + 1, dtype=np.float64)
    reach = THERMAL_CURL_LENGTH
    top = np.clip((reach - s) / reach, 0, None) ** 2
    foot = np.clip((s - (length - reach)) / reach, 0, None) ** 2
    angle = -np.radians(THERMAL_CURL[0]) * top + np.radians(THERMAL_CURL[1]) * foot

    def integrate(step):
        return np.concatenate([[0], np.cumsum((step[1:] + step[:-1]) / 2)])

    down, up = integrate(np.cos(angle)), integrate(np.sin(angle))
    middle = length // 2
    down += middle - down[middle]
    up += THERMAL_LIFT - up[middle]

    return s, down, up, angle


def soften(field, radius):
    """Low-pass a 2D array of floats: three box blurs each way, `radius`
    pixels on either side, as near a Gaussian as matters."""
    radius = max(1, round(radius))
    for axis in (0, 1):
        for _ in range(3):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius + 1, radius)
            total = np.cumsum(np.pad(field, pad, mode="edge"), axis=axis, dtype=np.float64)
            ahead = np.take(total, np.arange(2 * radius + 1, total.shape[axis]), axis=axis)
            behind = np.take(total, np.arange(0, total.shape[axis] - 2 * radius - 1), axis=axis)
            field = ((ahead - behind) / (2 * radius + 1)).astype(np.float32)

    return field


def wood_table(size, scale, rng):
    """Build the table's normal map, an array of floats: a top veneered in
    ash, stained black and finished satin, all alike, so the grain shows
    only in its relief. The open pores of each ring's early wood stay open
    under the finish, grooves in long fine dashes wandering with the rings,
    and the soft early wood has worn a little below the late."""
    w, h = size
    per_mm = scale * DOTS_PER_MM                    # pixels per millimeter

    def smooth(cells):
        noise = noise_image(cells, rng).resize(size, Image.BICUBIC)
        return np.asarray(noise, dtype=np.float32) / 255

    # The rings run across the image, their lines wandering: broad waves,
    # and smaller ones on them
    y = np.arange(h, dtype=np.float32)[:, None]
    flow = (y + SCENE_GRAIN_WAVE * per_mm * (smooth((4, 3)) - 0.5)
            + 0.3 * SCENE_GRAIN_WAVE * per_mm * (smooth((12, 9)) - 0.5))
    phase = flow / (SCENE_RING * per_mm)
    early = np.clip(1 - (phase % 1) / 0.45, 0, 1)      # each ring's porous band

    # The pores: dashes along the grain, laid on the rings' lines
    pores = noise_image((max(2, round(w / (SCENE_PORE_LENGTH * per_mm))),
                         max(2, round(h / (SCENE_PORE_WIDTH * per_mm)))), rng)
    pores = np.asarray(pores.resize(size, Image.BICUBIC), dtype=np.float32) / 255
    rows = np.clip(np.round(flow).astype(np.int32), 0, h - 1)
    pores = pores[rows, np.arange(w)[None, :]]
    open_pore = np.clip((pores - 0.86 + 0.2 * early) * 8, 0, 1)

    # The surface's height, in dots, then its slopes, and the normal they
    # tilt to, encoded in the normal map's usual 0 to 1
    height = -SCENE_PORE_DEPTH * open_pore - SCENE_RING_DEPTH * early
    height = soften(np.broadcast_to(height, (h, w)), SCENE_RELIEF_BLUR * per_mm)
    dy, dx = np.gradient(height * scale)
    normal = np.stack([-dx, dy, np.ones_like(dx)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    return normal * 0.5 + 0.5


def receipt_mesh(texture, rng):
    """Build the receipt's surface as a triangle mesh in the table's
    millimeters, x to the right of the image, y up it and z up off the
    table, with texture coordinates into the flat receipt; return its
    vertices, faces and texture coordinates."""
    width = texture.width / THERMAL_TEXTURE
    length = texture.height / THERMAL_TEXTURE
    s, down, up, _ = receipt_profile(round(length))
    center = (down[0] + down[-1]) / 2

    across, along = 81, len(s) // 2 + 1
    u, t = np.meshgrid(np.linspace(0, width, across), np.linspace(0, s[-1], along))
    v = np.interp(t, s, down)
    wave = noise_image((6, 7), rng).resize((across, along), Image.BICUBIC)
    z = (np.interp(t, s, up) + THERMAL_WAVE * (np.asarray(wave) / 255 - 0.5)
         + THERMAL_EDGE_LIFT * ((u - width / 2) / (width / 2)) ** 4)
    z = np.maximum(z, THERMAL_LIFT / 2)

    turn = math.radians(THERMAL_TILT)
    x0, y0 = u - width / 2, v - center
    x = math.cos(turn) * x0 - math.sin(turn) * y0
    y = -(math.sin(turn) * x0 + math.cos(turn) * y0)
    vertices = np.stack([x, y, z], -1).reshape(-1, 3) / DOTS_PER_MM
    texcoords = np.stack([u / width, t / s[-1]], -1).reshape(-1, 2)

    corner = (np.arange(along - 1)[:, None] * across + np.arange(across - 1)[None, :]).ravel()
    faces = np.concatenate([np.stack([corner, corner + across, corner + 1], -1),
                            np.stack([corner + 1, corner + across, corner + across + 1], -1)])

    return vertices.astype(np.float32), faces.astype(np.uint32), texcoords.astype(np.float32)


def render_receipt(texture, canvas_size, rng):
    """Lay the printed receipt on the table and path trace the scene with
    Mitsuba: the paper's soft shadows on the table and on itself, the light
    it throws back, the coating's faint gloss and the varnish's, and the
    lens's shallow depth of field all come of the light's own paths."""
    import mitsuba as mi
    mi.set_variant("scalar_rgb")

    def bitmap(pixels, srgb=True):
        bmp = mi.Bitmap(np.ascontiguousarray(pixels, dtype=np.float32))
        bmp.set_srgb_gamma(srgb)
        return bmp

    vertices, faces, texcoords = receipt_mesh(texture, rng)
    mesh = mi.Mesh("receipt", len(vertices), len(faces), has_vertex_texcoords=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = vertices.ravel()
    params["faces"] = faces.ravel()
    params["vertex_texcoords"] = texcoords.ravel()
    params.update()

    paper = np.asarray(texture, dtype=np.float32) / 255
    table_px = (SCENE_TABLE_TEXTURE,
                round(SCENE_TABLE_TEXTURE * SCENE_TABLE_SIZE[1] / SCENE_TABLE_SIZE[0]))
    relief = wood_table(table_px, table_px[0] / SCENE_TABLE_SIZE[0] / DOTS_PER_MM, rng)

    tilt = math.radians(SCENE_CAMERA_TILT)
    camera = [0, -SCENE_CAMERA_HEIGHT * math.sin(tilt), SCENE_CAMERA_HEIGHT * math.cos(tilt)]
    transform = mi.ScalarTransform4f

    with tempfile.TemporaryDirectory() as folder:
        ply = str(Path(folder) / "receipt.ply")
        mesh.write_ply(ply)
        scene = mi.load_dict({
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 8},
            "sensor": {
                "type": "thinlens",
                "aperture_radius": SCENE_APERTURE,
                "focus_distance": SCENE_CAMERA_HEIGHT,
                "fov": 2 * math.degrees(math.atan(SCENE_VIEW / 2 / SCENE_CAMERA_HEIGHT)),
                "fov_axis": "x",
                "to_world": transform().look_at(origin=camera, target=[0, 0, 0], up=[0, 1, 0]),
                "sampler": {"type": "independent", "sample_count": SCENE_SAMPLES,
                            "seed": NOISE_SEED},
                "film": {"type": "hdrfilm", "width": canvas_size[0], "height": canvas_size[1],
                         "rfilter": {"type": "gaussian"}},
            },
            "receipt": {
                "type": "ply", "filename": ply, "face_normals": False,
                "bsdf": {
                    "type": "mask",
                    "opacity": {"type": "bitmap", "bitmap": bitmap(paper[..., 3:], False),
                                "raw": True},
                    "bsdf": {"type": "twosided", "bsdf": {
                        "type": "principled",
                        "base_color": {"type": "bitmap", "bitmap": bitmap(paper[..., :3])},
                        "roughness": 0.45, "specular": 0.35}},
                },
            },
            "table": {
                "type": "rectangle",
                "to_world": transform().scale([SCENE_TABLE_SIZE[0] / 2, SCENE_TABLE_SIZE[1] / 2, 1]),
                "bsdf": {
                    "type": "normalmap",
                    "normalmap": {"type": "bitmap", "bitmap": bitmap(relief, False), "raw": True},
                    "bsdf": {"type": "principled", "specular": 0.4, "roughness": SCENE_FINISH,
                             "base_color": {"type": "rgb",
                                            "value": [(c / 255) ** 2.2 for c in SCENE_TABLE]}},
                },
            },
            "lamp": {
                "type": "rectangle",
                "to_world": transform().look_at(origin=SCENE_LAMP, target=[0, 0, 0], up=[0, 0, 1])
                            @ transform().scale([SCENE_LAMP_SIZE / 2, SCENE_LAMP_SIZE / 2, 1]),
                "emitter": {"type": "area",
                            "radiance": {"type": "rgb", "value": SCENE_LAMP_RADIANCE}},
            },
            "room": {"type": "constant",
                     "radiance": {"type": "rgb", "value": SCENE_ROOM_RADIANCE}},
        })
        img = mi.render(scene)

    img = Image.fromarray(np.array(mi.util.convert_to_bitmap(img)))

    # The lens's vignette: the light falls off smoothly from the middle,
    # measured out to the corners
    aspect = img.width / img.height

    def vignette(u, v):
        r = ((u - 0.5) ** 2 * aspect ** 2 + (v - 0.5) ** 2) / (0.25 * aspect ** 2 + 0.25)
        return round(255 * (1 - SCENE_VIGNETTE * r ** 1.2))

    return shade(img, light_field(img.size, vignette))


def build_ramp():
    """The size ramp, on a thermal receipt lying on a table."""
    rng = random.Random(NOISE_SEED)
    texture = print_receipt(compose_receipt(THERMAL_LENGTH), rng)
    save_image(render_receipt(texture, CANVAS, rng), "sample3")


# --- Paper -----------------------------------------------------------------

# Every printer runs the same stock
PAPER_COLOR = (250, 249, 246)
PAPER_GRAIN = 9                 # depth of the paper grain, of 255


# --- Inkjet: the variation axes proof ---------------------------------------

INKJET_DOT = 4                  # Printer dot pitch, in image pixels
INKJET_INK = (36, 40, 52)       # Dye-based black: slightly weak and bluish
INKJET_SWATH = 50               # Nozzles per print head pass
INKJET_GAIN = (0.56, 0.70)      # Droplet radius range, in dots (dot gain > 0.5)
INKJET_SATELLITES = 0.03        # Chance of a stray satellite drop per edge dot

# The proof card, as Tiny5's: a running head over three rows of two blocks,
# each block an axis name with its technical tag beneath. The card is
# trimmed to a wide side margin and a narrower one head and foot, and the
# blocks stand in two equal columns across the measure it leaves
PROOF_MARGIN_X = 181            # side margin, in image pixels
PROOF_MARGIN_Y = 84             # head and foot margin, in image pixels
PROOF_WORD_SCALE = 20           # the axis name, in image pixels per font
                                # pixel down: Mono sets wider than Tiny5, and
                                # "roundness" must still fit its column
PROOF_TAG_SCALE = 8             # its tag, and the running head
PROOF_TAG_GAP = 30              # ink gap from an axis name down to its tag

# Each variation axis demonstrated by its own name, set with that axis
# pushed to its extreme
AXIS_ROWS = [
    ("weight", "wght 900", {"axes": (900, 100, 0, 0, 0)}),
    ("width", "wdth 75", {"axes": (300, 75, 0, 0, 0)}),
    ("italic", "ital 1", {"axes": (300, 100, 0, 0, 0), "italic": True}),
    ("roundness", "ROND 100", {"axes": (300, 100, 100, 0, 0)}),
    ("bleed", "BLED 80", {"axes": (200, 100, 0, 80, 0)}),
    ("jitter", "JITT 100", {"axes": (300, 100, 0, 0, 100)}),
]


@lru_cache(maxsize=None)
def proof_font(scale, axes=REGULAR_AXES, italic=False):
    """Load Tiny5 Mono sized so one font pixel is `scale` image pixels tall,
    at the given axis settings."""
    path = MONO_ITALIC_PATH if italic else MONO_PATH
    font = ImageFont.truetype(font=str(path), size=8 * scale)
    font.set_variation_by_axes(list(axes))

    return font


def render_inkjet(mask, rng):
    """Print a full-resolution ink mask as an early inkjet would: rasterized
    to the printer's dot grid, each dot an overgrown droplet placed with a
    little error and the odd satellite, feathered into the paper fibers,
    with faint banding where the head passes meet."""
    size = mask.size
    dot = INKJET_DOT
    grid = (size[0] // dot, size[1] // dot)
    dots = mask.resize(grid, Image.BOX).point(lambda v: 255 if v >= 128 else 0)
    px = dots.load()

    ink = Image.new("L", size, 0)
    draw = ImageDraw.Draw(ink)
    lo, hi = INKJET_GAIN
    pass_offset = 0
    for y in range(grid[1]):
        # Each swath is a separate pass: a small vertical registration
        # error, and the last nozzle rows lay down a little less ink
        if y % INKJET_SWATH == 0:
            pass_offset = rng.uniform(-0.6, 0.6)
        row_level = 255 if y % INKJET_SWATH < INKJET_SWATH - 2 else 215
        for x in range(grid[0]):
            if not px[x, y]:
                continue
            cx = (x + 0.5) * dot + rng.uniform(-0.5, 0.5)
            cy = (y + 0.5) * dot + pass_offset + rng.uniform(-0.4, 0.4)
            r = rng.uniform(lo, hi) * dot
            draw.ellipse(xy=[cx - r, cy - r, cx + r, cy + r], fill=row_level)
            # Satellites spray off the edge dots
            edge = (x == 0 or y == 0 or x == grid[0] - 1 or y == grid[1] - 1 or
                    not (px[x - 1, y] and px[x + 1, y] and px[x, y - 1] and px[x, y + 1]))
            if edge and rng.random() < INKJET_SATELLITES:
                sx = cx + rng.uniform(-2.5, 2.5) * dot
                sy = cy + rng.uniform(-1.0, 1.0) * dot
                sr = rng.uniform(0.15, 0.3) * dot
                draw.ellipse(xy=[sx - sr, sy - sr, sx + sr, sy + sr], fill=170)

    # Ink wicks along the paper fibers: soften the droplets, then re-harden
    # the edge with a little correlated noise so the outline turns fibrous
    ink = ink.filter(ImageFilter.GaussianBlur(dot * 0.25))
    fibers = noise_image(size, rng).filter(ImageFilter.GaussianBlur(1.6))
    fibers = fibers.point(lambda v: 128 + (v - 128) * 1.8)
    ink = ImageChops.add(ink, fibers, scale=1.0, offset=-128)
    ink = ink.point(lambda v: min(255, max(0, (v - 128) * 4 + 128)))
    ink = ink.filter(ImageFilter.GaussianBlur(0.5))

    # Uneven absorption: the ink density drifts a little across the page
    density = noise_image((size[0] // 96 + 1, size[1] // 96 + 1), rng)
    density = density.resize(size, Image.BICUBIC).point(lambda v: 222 + v * 33 // 255)
    ink = ImageChops.multiply(ink, density)

    paper = shade(Image.new("RGB", size, PAPER_COLOR),
                  grain_field(size, rng, PAPER_GRAIN))

    return Image.composite(Image.new("RGB", size, INKJET_INK), paper, ink)


def build_axes():
    """A proof card for the variation axes, run off on an early inkjet: each
    axis shown by its own name, typeset with that axis at its extreme, drawn
    directly at a large size so the font's own axis effects reproduce
    faithfully, then rasterized to the printer's dots.

    The card is spaced on the ink rather than on the metrics: the axes push
    the glyphs to their own heights, so blocks stepped by a fixed baseline
    pitch would leave visibly uneven gaps. Each tag instead clears the
    descenders of its own row, and the space left over spreads evenly.

    Across the page it is set the same way. Bleed swells the ink past the
    pen and italic and jitter lean and shift it, so lines placed by their
    pens would hang off the column by a fraction of a font pixel: every
    line is set against its own ink instead.
    """
    rng = random.Random(NOISE_SEED)
    mask = Image.new("L", CANVAS, 0)
    draw = ImageDraw.Draw(mask)

    def ink_box(scale, string, **variant):
        """Return the box the string's ink covers, relative to its pen,
        measured by setting the string on its own."""
        font = proof_font(scale, **variant)
        pad = 2 * scale          # room for the ink that falls outside the pen
        width = round(draw.textlength(string, font=font)) + 2 * pad
        height = 2 * (font.size + pad)
        proof = Image.new("L", (width, height), 0)
        ImageDraw.Draw(proof).text(xy=(pad, height // 2), text=string, fill=255,
                                   font=font, anchor="ls")
        box = proof.getbbox()

        return (box[0] - pad, box[1] - height // 2,
                box[2] - pad, box[3] - height // 2)

    def put(xy, scale, string, anchor="ls", **variant):
        """Draw a string set against x by its ink rather than by its pen, so
        that a column edge, and the running head, line up on what shows."""
        box = ink_box(scale, string, **variant)
        x = xy[0] - (box[2] if anchor[0] == "r" else box[0])
        draw.text(xy=(x, xy[1]), text=string, fill=255,
                  font=proof_font(scale, **variant), anchor="l" + anchor[1])

    def ink(scale, strings):
        """Return how far the tallest of the strings inks above the baseline,
        and the deepest of them below it."""
        boxes = [ink_box(scale, string, **variant) for string, variant in strings]

        return max(-box[1] for box in boxes), max(box[3] for box in boxes)

    left, right = PROOF_MARGIN_X, CANVAS[0] - PROOF_MARGIN_X
    column = (right - left) // 2        # pitch of the two columns
    title = "Tiny5 Mono variation test"
    head = [(title, {}), (VERSION, {})]
    rows = [AXIS_ROWS[i:i + 2] for i in range(0, len(AXIS_ROWS), 2)]

    # Measure every line, then share out what the blocks leave: one gap
    # above each row, all equal
    head_above, head_below = ink(PROOF_TAG_SCALE, head)
    blocks = []
    for row in rows:
        above, below = ink(PROOF_WORD_SCALE, [(word, v) for word, _, v in row])
        tag_above, tag_below = ink(PROOF_TAG_SCALE, [(tag, {}) for _, tag, _ in row])
        blocks.append((above, below + PROOF_TAG_GAP + tag_above, tag_below))
    filled = head_above + head_below + sum(sum(block) for block in blocks)
    gap = (CANVAS[1] - 2 * PROOF_MARGIN_Y - filled) // len(rows)

    y = PROOF_MARGIN_Y + head_above
    put((left, y), PROOF_TAG_SCALE, title)
    put((right, y), PROOF_TAG_SCALE, VERSION, anchor="rs")
    y += head_below

    for row, (above, tag_offset, tag_below) in zip(rows, blocks):
        y += gap + above
        for index, (word, tag, variant) in enumerate(row):
            x = left + index * column
            put((x, y), PROOF_WORD_SCALE, word, **variant)
            put((x, y + tag_offset), PROOF_TAG_SCALE, tag)
        y += tag_offset + tag_below

    save_image(render_inkjet(mask, rng), "inkjet")


# --- Variation axes: the animation -----------------------------------------

# The variation axes, animated as type designers show them: no device, the
# font itself on the editor's dark ground, as large as Tiny5's own
# presentation sets it. The name, and under it the subtitle, take every
# axis; beneath, a readout lists every axis by name with its value, three
# to a row, lit while it is off its default. Each axis in turn swings out
# to its extremes, each segment morphing straight out of the last, and it
# comes back to Regular, where it starts, so it loops.
ANIM_TITLE = ["Tiny5", "Mono"]                  # stacked, so the name can be large
ANIM_TITLE_PITCH = 8                            # font pixels from one line of it to the next
ANIM_SUBTITLE = "A 5-pixel font"
ANIM_CURSOR_BLINK = 1.0                         # seconds per blink of the cursor after it
ANIM_DIM = (88, 88, 88)                         # the readout's idle axes
ANIM_READOUT_COLUMNS = 3
ANIM_READOUT_GAP = 3                            # cells between the readout's columns
ANIM_SUPER_LINES = 4320                         # lines every frame is drawn at
ANIM_POSTER = 10.0                              # the still, at this time: round pixels

# The axes, as the readout lists them: tag, name, minimum and maximum
ANIM_AXES = [
    ("wght", "Weight", 100, 900),
    ("wdth", "Width", 50, 200),
    ("ital", "Italic", 0, 1),
    ("ROND", "Roundness", 0, 100),
    ("BLED", "Bleed", 0, 100),
    ("JITT", "Jitter", 0, 100),
]
ANIM_REGULAR = {"wght": 400, "wdth": 100, "ital": 0, "ROND": 0, "BLED": 0, "JITT": 0}

# The timeline: one segment to an axis, then one back to Regular, all of
# one length. A segment eases in steps through its keys, each a style, or
# the axes it changes from the last key; it starts where the last segment
# ended. Italic is a font of its own, not a point on an axis, and snaps
# over halfway through its step. Bleed and jitter show at their best on
# round pixels, so roundness swings up, down and up again and stays up;
# jitter then swings up and down. Once weight has shown its range it
# settles at 250, where the pixels read most clearly at this size, and
# stays there until the last segment eases every axis back to Regular, and
# rests there before the loop starts again
ANIM_STYLES = {
    "Regular": {},
}
ANIM_START = "Regular"
ANIM_SEGMENT = 2.5                              # seconds, every segment
ANIM_TIMELINE = [
    [{"wght": 900}, {"wght": 100}, {"wght": 250}],
    [{"wdth": 200}, {"wdth": 50}, {"wdth": 100}],
    [{"ital": 1}, {"ital": 0}],
    [{"ROND": 100}, {"ROND": 0}, {"ROND": 100}],
    [{"BLED": 100}, {"BLED": 0}],
    [{"JITT": 100}, {"JITT": 0}],
    ["Regular", {}],
]


# The layouts, for a frame of 1080 lines: the frame's width, the image
# pixels per font pixel of the name, the subtitle and the readout, and the
# baselines of the name's first line, the subtitle and the readout's first
# row. The name is as large as the frame's height allows, and the name and
# the subtitle still fit across at Width 200. Other heights scale the
# layout in proportion
ANIM_LAYOUTS = {
    "wide": {"width": 1920, "title": 36, "subtitle": 14, "readout": 6,
             "baselines": (306, 768, 892)},
    "square": {"width": 1080, "title": 24, "subtitle": 8, "readout": 5,
               "baselines": (373, 673, 779)},
}


@lru_cache(maxsize=64)
def anim_font(px, wght=400, wdth=100, rond=0, bled=0, jitt=0, italic=False):
    """Load Tiny5 Mono at `px` image pixels per font pixel, at the given
    axis settings."""
    font = ImageFont.truetype(str(MONO_ITALIC_PATH if italic else MONO_PATH), size=8 * px)
    font.set_variation_by_axes([wght, wdth, rond, bled, jitt])

    return font


def anim_state(time):
    """Return the axis values at a time in the animation, in seconds."""
    def ease(t):
        # Smootherstep: speed and acceleration both start and end at zero
        return t * t * t * (t * (6 * t - 15) + 10)

    def key_state(before, key):
        return dict(ANIM_REGULAR, **ANIM_STYLES[key]) if isinstance(key, str) else dict(before, **key)

    state = key_state(None, ANIM_START)
    time %= len(ANIM_TIMELINE) * ANIM_SEGMENT
    segment, t = divmod(time / ANIM_SEGMENT, 1)
    for keys in ANIM_TIMELINE[:int(segment)]:
        for key in keys:
            state = key_state(state, key)
    keys = ANIM_TIMELINE[int(segment)]
    step, t = divmod(t * len(keys), 1)
    for key in keys[:int(step)]:
        state = key_state(state, key)
    after = key_state(state, keys[int(step)])
    e = ease(t)

    return {tag: state[tag] + (after[tag] - state[tag]) * e for tag in state}


def anim_frame(state, cursor, layout, height):
    """Draw one frame: the name and the subtitle at the axis values, the
    cursor after it on or off, and the readout with the axes off their
    defaults lit."""
    k = height / 1080
    size = (round(layout["width"] * k), height)
    img = Image.new("RGB", size, ED_BG)
    draw = ImageDraw.Draw(img)
    axes = (state["wght"], state["wdth"], state["ROND"], state["BLED"], state["JITT"],
            state["ital"] >= 0.5)
    title_y, subtitle_y, readout_y = (round(y * k) for y in layout["baselines"])
    middle = size[0] / 2

    px = round(layout["title"] * k)
    for i, text in enumerate(ANIM_TITLE):
        draw.text((middle, title_y + i * ANIM_TITLE_PITCH * px), text, font=anim_font(px, *axes),
                  fill=ED_TEXT, anchor="ms")

    # The subtitle, centered whole, cursor and all, so the blink never
    # moves it
    font = anim_font(round(layout["subtitle"] * k), *axes)
    x = middle - draw.textlength(ANIM_SUBTITLE + "█", font=font) / 2
    draw.text((x, subtitle_y), ANIM_SUBTITLE + ("█" if cursor else ""), font=font, fill=ED_TEXT,
              anchor="ls")

    # The readout: each axis by name, its value set flush right, in a grid
    # that keeps its columns, as the font does; lit while off its default
    px = round(layout["readout"] * k)
    font = anim_font(px)
    label = max(len(name) for _, name, _, _ in ANIM_AXES) + 4
    advance = draw.textlength("M", font=font)
    pitch = (label + ANIM_READOUT_GAP) * advance
    left = middle - (ANIM_READOUT_COLUMNS * pitch - ANIM_READOUT_GAP * advance) / 2
    line = round(1.6 * CELL_HEIGHT * px)
    for i, (tag, name, _, _) in enumerate(ANIM_AXES):
        row, col = divmod(i, ANIM_READOUT_COLUMNS)
        value = round(state[tag])
        lit = value != ANIM_REGULAR[tag]
        text = f"{name:<{label - 4}}{value:>4}"
        draw.text((left + col * pitch, readout_y + row * line), text, font=font,
                  fill=ED_KEYWORD if lit else ANIM_DIM, anchor="ls")

    return img


def render_animation(name, layout, height, fps, encoder):
    """Render the animation frame by frame straight into ffmpeg, to
    tiny5mono-<name>, with the encoder's arguments."""
    check_coverage(set("".join(ANIM_TITLE) + ANIM_SUBTITLE + "█"))
    width = round(ANIM_LAYOUTS[layout]["width"] * height / 1080)
    length = len(ANIM_TIMELINE) * ANIM_SEGMENT
    # Each frame is drawn at a whole multiple of its size and reduced, so
    # the type moves smoothly between pixels rather than hopping from one
    # to the next, and its font pixels stay crisp
    over = max(1, ANIM_SUPER_LINES // height)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *encoder,
         str(OUT_PATH / f"tiny5mono-{name}")],
        stdin=subprocess.PIPE)
    for i in range(round(length * fps)):
        time = i / fps
        cursor = time % ANIM_CURSOR_BLINK < ANIM_CURSOR_BLINK / 2
        frame = anim_frame(anim_state(time), cursor, ANIM_LAYOUTS[layout], height * over)
        frame = frame.reduce(over)
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg failed on tiny5mono-{name}")


H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-movflags", "+faststart"]


def build_animation():
    """The variation axes, animated, as the presentation: a still for where
    only an image goes, the round pixels held; for the Google Fonts article
    an MP4 of at most 1.4 MB at its image size; a 4K master, WebM and
    GIF for the web, and a square cut for social media."""
    over = ANIM_SUPER_LINES // CANVAS[1]
    save_image(anim_frame(anim_state(ANIM_POSTER), True, ANIM_LAYOUTS["wide"],
                          CANVAS[1] * over).reduce(over), "presentation")
    render_animation("presentation.mp4", "wide", 1080, 30, [*H264, "-crf", "22"])
    render_animation("presentation-google.mp4", "wide", GOOGLE_SIZE[1], 30,
                     [*H264, "-crf", "22"])
    render_animation("axes-4k.mp4", "wide", 2160, 60, [*H264, "-crf", "16"])
    render_animation("axes.webm", "wide", 1080, 30,
                     ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34", "-row-mt", "1"])
    render_animation("axes.gif", "wide", 540, 15,
                     ["-vf", "split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];"
                             "[b][p]paletteuse=dither=none"])
    render_animation("axes-square.mp4", "square", 1080, 30, [*H264, "-crf", "21"])


# --- Modern terminal: the shell session ------------------------------------

# A terminal window, maximized, its tab bar along the top edge: a shell
# session that fetches the font's facts beside its name in sextants, then
# lists a directory. No device: the text is the font itself, drawn at a
# light weight with some bleed, which thins every pixel to a bar and joins
# the bars across, so the type carries scanlines of its own
TERM_PX = 5                     # image pixels per font pixel
TERM_AXES = (300, 100, 0, 50, 0)            # wght, wdth, ROND, BLED, JITT
TERM_OVER = 4                   # the text is drawn this many times over size
TERM_TAB_BAR = 52               # the tab bar's height, in image pixels
TERM_PAD = 20                   # the terminal's padding, in image pixels

# VS Code's terminal colors, to match the hero
TERM_BG = (30, 30, 30)
TERM_FG = (204, 204, 204)
TERM_BAR = (37, 37, 38)
TERM_TAB_TEXT = (220, 220, 220)
TERM_TAB_DIM = (150, 150, 150)
TERM_NORMAL = [(0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16),
               (36, 114, 200), (188, 63, 188), (17, 168, 205), (229, 229, 229)]
TERM_BRIGHT = [(102, 102, 102), (241, 76, 76), (35, 209, 139), (245, 245, 67),
               (59, 142, 234), (214, 112, 214), (41, 184, 219), (229, 229, 229)]
TERM_RED, TERM_GREEN, TERM_YELLOW = TERM_BRIGHT[1:4]
TERM_BLUE, TERM_CYAN, TERM_GRAY = TERM_BRIGHT[4], TERM_BRIGHT[6], TERM_BRIGHT[0]

TERM_FACTS = [
    ("Font", "Tiny5 Mono " + VERSION[1:]),
    ("Cell", "6 × 9 pixels"),
    ("Pixel", "89:128"),
    ("Axes", "wght wdth ital ROND BLED JITT"),
    ("Glyphs", "2,287"),
    ("Languages", "974"),
    ("License", "OFL-1.1"),
]
TERM_LISTING = [
    ("drwxr-xr-x", " 4 user user  4096 Oct  7 12:00 ", "documentation", TERM_BLUE),
    ("drwxr-xr-x", " 3 user user  4096 Oct  7 12:00 ", "fonts", TERM_BLUE),
    ("drwxr-xr-x", " 2 user user  4096 Oct  7 12:00 ", "sources", TERM_BLUE),
    ("-rw-r--r--", " 1 user user  4398 Oct  7 12:00 ", "OFL.txt", TERM_FG),
    ("-rwxr-xr-x", " 1 user user  1207 Oct  7 12:00 ", "build.sh", TERM_GREEN),
]


def compose_terminal(cols, rows):
    """Compose the session: the fetch, the name in sextants beside the
    facts and the color swatches, then a long listing, and the prompt
    waiting with its cursor."""
    s = TextScreen(cols, rows, TERM_FG, TERM_BG)

    def prompt(row, command):
        col = 0
        for text, color in [("user@tiny5", TERM_GREEN), (":", TERM_FG),
                            ("~/tiny5mono", TERM_BLUE), ("$ ", TERM_FG), (command, TERM_FG)]:
            s.put(col, row, text, color)
            col += len(text)
        return col

    prompt(0, "tiny5fetch")
    s.big_text(2, 2, "Tiny5", TERM_CYAN, scale=2, clear=False)
    s.big_text(2, 8, "Mono", TERM_CYAN, scale=2, clear=False)
    s.put(36, 2, "user", TERM_CYAN)
    s.put(40, 2, "@", TERM_FG)
    s.put(41, 2, "tiny5", TERM_CYAN)
    s.put(36, 3, "─" * 10, TERM_FG)
    for row, (label, value) in enumerate(TERM_FACTS, 4):
        s.put(36, row, label, TERM_YELLOW)
        s.put(36 + len(label), row, ": " + value, TERM_FG)
    for i, (normal, bright) in enumerate(zip(TERM_NORMAL, TERM_BRIGHT)):
        s.put(36 + 3 * i, 12, "███", normal)
        s.put(36 + 3 * i, 13, "███", bright)

    # ls colors each permission by its kind
    permission = {"d": TERM_BLUE, "r": TERM_YELLOW, "w": TERM_RED, "x": TERM_GREEN,
                  "-": TERM_GRAY}
    prompt(15, "ls -l")
    for row, (mode, middle, name, color) in enumerate(TERM_LISTING, 16):
        for col, flag in enumerate(mode):
            s.put(col, row, flag, permission[flag])
        s.put(len(mode), row, middle, TERM_FG)
        s.put(len(mode) + len(middle), row, name, color)
    s.put(prompt(21, ""), 21, "█", TERM_FG)

    return s


@lru_cache(maxsize=None)
def term_font(px, axes=TERM_AXES):
    """Load Tiny5 Mono at `px` image pixels per font pixel."""
    font = ImageFont.truetype(str(MONO_PATH), size=8 * px)
    font.set_variation_by_axes(list(axes))

    return font


def render_terminal_text(screen):
    """Render a screen with the variable font itself, so the axes shape
    every glyph, drawn TERM_OVER times over size and reduced: each
    character set at its own cell, so the columns never drift."""
    check_coverage({cell[0] for line in screen.cells for cell in line})
    px = TERM_PX * TERM_OVER
    font = term_font(px)
    advance = font.getlength("M")
    line = CELL_HEIGHT * px
    img = Image.new("RGB", (round(screen.cols * advance), screen.rows * line), TERM_BG)
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(screen.cells):
        for x, (char, fg, _, _) in enumerate(row):
            if char != " ":
                draw.text((round(x * advance), y * line + CELL_ASCENT * px), char, font=font,
                          fill=fg, anchor="ls")

    return img.reduce(TERM_OVER)


def render_terminal(canvas_size):
    """Lay the terminal out over the whole frame: the tab bar with its one
    tab, then as many cells as the window holds."""
    advance = term_font(TERM_PX * TERM_OVER).getlength("M") / TERM_OVER
    cols = int((canvas_size[0] - 2 * TERM_PAD) / advance)
    rows = int((canvas_size[1] - TERM_TAB_BAR - TERM_PAD) / (CELL_HEIGHT * TERM_PX))

    img = Image.new("RGB", canvas_size, TERM_BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, canvas_size[0], TERM_TAB_BAR], fill=TERM_BAR)
    tab = 260
    draw.rectangle([0, 0, tab, TERM_TAB_BAR], fill=TERM_BG)
    draw.line([0, 0, tab, 0], fill=TERM_BLUE, width=3)      # the active tab's accent
    font = term_font(3, REGULAR_AXES)
    baseline = TERM_TAB_BAR / 2 + 8
    for x, text, color in [(24, ">_", TERM_GREEN), (70, "bash", TERM_TAB_TEXT),
                           (tab - 34, "×", TERM_TAB_DIM), (tab + 26, "+", TERM_TAB_DIM)]:
        draw.text((x, baseline), text, font=font, fill=color, anchor="ls")

    img.paste(render_terminal_text(compose_terminal(cols, rows)),
              (TERM_PAD, TERM_TAB_BAR + TERM_PAD // 2))

    return img


def build_terminal():
    """The shell session, in a modern terminal."""
    save_image(render_terminal(CANVAS), "terminal")


# --- 9-pin dot matrix: the report -----------------------------------------

# A report off a 9-pin dot-matrix printer, on continuous form paper, as
# Tiny5's self test: the job a monospace font was made for, figures standing
# in their columns. The head's 9 pins strike a whole character cell in one
# pass, descenders and all, and it steps closer across than it feeds down,
# so each font pixel is one dot, set 89:128. The report is printed
# single-spaced; the paper's green bars are two lines tall, and the page
# fits the report, its margin above and below
MATRIX_MARGIN = 4               # paper above and below the report, in font pixels
MATRIX_SUPER = 3                # the ink is struck this many times over size
MATRIX_INDENT = 3               # cells from the tractor margin to the text
MATRIX_BAR = (216, 236, 228)
MATRIX_INK = (38, 35, 42)
MATRIX_DOT = (0.62, 0.56)       # a dot's radius, in dot pitches across and down
MATRIX_DOT_JITTER = (0.11, 0.09)  # how far a dot lands off its place, likewise
MATRIX_SPREAD = (1.0, 0.7)      # ink soaking into the paper, smeared by the head
                                # travel, in image pixels
MATRIX_SKEW = 0.15              # the paper feeds in askew, in degrees
MATRIX_PASS_OFFSET = 0.3        # the head lands off between passes, in dot pitches
MATRIX_PATCH = 24               # the ribbon inks in patches, in font pixels
MATRIX_PATCH_LEVEL = 165        # ink level of the faintest patch, of 255
MATRIX_WEAK_PINS = 2            # print head pins that strike faintly

# The branches, each with its units, revenue and change on last quarter,
# sorted as a computer sorts them, by code point: Greek and Cyrillic last.
# The totals, the change overall and the shares are worked out from these
REPORT_TITLE = "SALES BY BRANCH, Q3"
REPORT_BRANCHES = [
    ("Győr", 712, 1281600, 6.4),
    ("Hà Nội", 1046, 1569000, 18.2),
    ("İstanbul", 1318, 2240600, 9.7),
    ("Kraków", 932, 1677600, 11.8),
    ("München", 1284, 2311200, 4.2),
    ("Reykjavík", 268, 589600, -2.1),
    ("São Paulo", 1105, 1878500, 7.5),
    ("Αθήνα", 547, 984600, -0.8),
    ("София", 623, 1059100, 3.3),
]
REPORT_SHARE_CELLS = 12         # the longest share bar, in cells


def share_bar(fraction, cells):
    """Draw a bar `fraction` of `cells` long, to half a cell, in sextants:
    their middle row, which stands level with the capitals, so the bars
    of one row and the next keep apart."""
    halves = round(fraction * cells * 2)

    return sextant(12) * (halves // 2) + sextant(4) * (halves % 2)


def compose_report():
    """Compose the report as the printer gets it: a running head with the
    title and the page, the column heads ruled off, a row to a branch, its
    revenue drawn as a bar against the largest, and the totals under a double
    rule. Returns the lines."""
    def row(branch, units, revenue, change, bar=""):
        return f"{branch:<10}{units:>7,}{revenue:>12,}{change:>+8.1f}%  {bar}"

    top = max(revenue for _, _, revenue, _ in REPORT_BRANCHES)
    rows = [row(*branch, share_bar(branch[2] / top, REPORT_SHARE_CELLS))
            for branch in REPORT_BRANCHES]
    units = sum(branch[1] for branch in REPORT_BRANCHES)
    revenue = sum(branch[2] for branch in REPORT_BRANCHES)
    before = sum(branch[2] / (1 + branch[3] / 100) for branch in REPORT_BRANCHES)
    total = row("TOTAL", units, revenue, 100 * (revenue / before - 1))

    head = f"{'BRANCH':<10}{'UNITS':>7}{'REVENUE':>12}{'CHANGE':>9}  SHARE"
    width = len(head) - len("SHARE") + REPORT_SHARE_CELLS

    return ([REPORT_TITLE + "PAGE 1".rjust(width - len(REPORT_TITLE)), "", head, "─" * width]
            + rows + ["═" * width, total])


def compose_printout(lines):
    """Return the mask of the dots to strike for some lines of text, one per
    font pixel."""
    check_coverage({char for line in lines for char in line})
    width = max(len(line) for line in lines)
    mask = Image.new("L", (width * CELL_WIDTH, len(lines) * CELL_HEIGHT), 0)
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if char != " ":
                mask.paste(glyph_cell(char), (col * CELL_WIDTH, row * CELL_HEIGHT))

    return mask


def form_paper(size, pitch, top, rng):
    """Draw continuous form paper: green bars two print lines tall, from the
    printout's first line on, the perforated tractor margin with its
    sprocket holes, and paper grain."""
    img = Image.new("RGB", size, PAPER_COLOR)
    draw = ImageDraw.Draw(img)
    line = CELL_HEIGHT * pitch
    strip = round(2 * line)
    for i in range(-4, math.ceil((size[1] - top) / line), 4):
        draw.rectangle(xy=[(strip, top + i * line), (size[0], top + (i + 2) * line - 1)],
                       fill=MATRIX_BAR)

    # The tractor margin, creased and perforated
    draw.rectangle(xy=[(strip - 2, 0), (strip + 1, size[1])], fill=(230, 230, 227))
    for y in range(0, size[1], round(4 * pitch)):
        draw.rectangle(xy=[(strip - 2, y), (strip + 1, y + round(2 * pitch))],
                       fill=(202, 202, 199))

    # Sprocket holes, punched through to the dark platen
    radius = 2 * pitch
    y = top + line % (2 * line) - 2 * line
    while y < size[1] + radius:
        xy = (strip / 2, y)
        draw.circle(xy=xy, radius=radius + 2, fill=(208, 208, 205))
        draw.circle(xy=xy, radius=radius, fill=(28, 26, 30))
        y += 2 * line

    return shade(img, grain_field(size, rng, PAPER_GRAIN))


def strike_printout(mask, canvas_size, rng):
    """Strike every dot of a printout onto the paper, each slightly off in
    place, size and ink. The ribbon inks in patches, worn pins strike
    faintly, the head lands slightly off between passes, and the paper
    feeds in a little askew."""
    pitch_y = canvas_size[1] / (mask.height + 2 * MATRIX_MARGIN)
    pitch_x = pitch_y * PIXEL_ASPECT
    top = MATRIX_MARGIN * pitch_y
    left = 2 * CELL_HEIGHT * pitch_y + MATRIX_INDENT * CELL_WIDTH * pitch_x

    patches = noise_image((max(mask.width // MATRIX_PATCH, 1), max(mask.height // MATRIX_PATCH, 1)),
                          rng).resize(mask.size, Image.BICUBIC)
    patches = patches.point(lambda v: MATRIX_PATCH_LEVEL + v * (255 - MATRIX_PATCH_LEVEL) // 255).load()
    weak_pins = {pin: rng.randrange(150, 230)
                 for pin in rng.sample(range(CELL_HEIGHT), MATRIX_WEAK_PINS)}

    s = MATRIX_SUPER
    ink = Image.new("L", (canvas_size[0] * s, canvas_size[1] * s), 0)
    draw = ImageDraw.Draw(ink)
    dots = mask.load()
    for py in range(mask.height):
        line, pin = divmod(py, CELL_HEIGHT)
        offset = MATRIX_PASS_OFFSET * pitch_x * (line % 2)
        for px in range(mask.width):
            if not dots[px, py]:
                continue
            level = patches[px, py] * weak_pins.get(pin, 255) // 255
            x = left + (px + 0.5) * pitch_x + offset + rng.uniform(-1, 1) * MATRIX_DOT_JITTER[0] * pitch_x
            y = top + (py + 0.5) * pitch_y + rng.uniform(-1, 1) * MATRIX_DOT_JITTER[1] * pitch_y
            size = rng.uniform(0.92, 1.05)
            rx, ry = MATRIX_DOT[0] * pitch_x * size, MATRIX_DOT[1] * pitch_y * size
            draw.ellipse(xy=[(x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s], fill=level)
    ink = ink.filter(ImageFilter.GaussianBlur((MATRIX_SPREAD[0] * s, MATRIX_SPREAD[1] * s)))
    ink = ink.rotate(MATRIX_SKEW, resample=Image.BILINEAR).reduce(s)

    img = form_paper(canvas_size, pitch_y, top, rng)
    img.paste(Image.new("RGB", canvas_size, MATRIX_INK), (0, 0), ink)

    return img


def build_report():
    """The report, off a 9-pin printer."""
    rng = random.Random(NOISE_SEED)
    save_image(strike_printout(compose_printout(compose_report()), CANVAS, rng), "sample5")


if __name__ == "__main__":
    build_animation()                           # the presentation: variation axes, animated
    build_editor()                              # 1: code, modern editor on a color CRT
    build_charset()                             # 2: character set, character LCD
    build_ramp()                                # 3: size ramp, thermal receipt
    build_monitor()                             # 4: system monitor, color CRT
    build_report()                              # 5: sales report, 9-pin printer
