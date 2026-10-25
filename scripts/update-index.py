#!/usr/bin/env python3
# Updates the glyph table of the GitHub Pages glyph viewer (index.html) from
# the variable fonts. The table sits between the BEGIN GLYPHS and END GLYPHS
# markers; everything else in the page is edited by hand.
#
# Glyphs are ordered for finding, not by code point: letters by script and
# case, each base letter followed by its variants (accented, stroked, turned,
# ...), then numbers, punctuation and the symbol groups.
import argparse
import json
import os
import re
import urllib.parse

from fontTools import unicodedata as fontunicodedata
from fontTools.ttLib import TTFont

try:
    import unicodedata2 as unicodedata
except ImportError:
    import unicodedata

AXIS_ORDER = ["wght", "wdth", "ital", "ROND", "BLED", "JITT"]

SCRIPTS = [
    ("Latn", "Latin"),
    ("Grek", "Greek"),
    ("Cyrl", "Cyrillic"),
    ("Armn", "Armenian"),
]

# First letter and last letter of each script's basic alphabet: base letters in
# these ranges come first, in code point order.
CORE_LETTERS = {
    "Latn": [(0x41, 0x5A), (0x61, 0x7A)],
    "Grek": [(0x391, 0x3A9), (0x3B1, 0x3C9)],
    "Cyrl": [(0x410, 0x42F), (0x430, 0x44F)],
    "Armn": [(0x531, 0x556), (0x561, 0x586)],
}

# Unnamed private use characters
POWERLINE_NAMES = {
    0xE0A0: "POWERLINE BRANCH",
    0xE0A1: "POWERLINE LINE NUMBER",
    0xE0A2: "POWERLINE READONLY",
    0xE0A3: "POWERLINE COLUMN NUMBER",
    0xE0B0: "POWERLINE LEFT HARD DIVIDER",
    0xE0B1: "POWERLINE LEFT SOFT DIVIDER",
    0xE0B2: "POWERLINE RIGHT HARD DIVIDER",
    0xE0B3: "POWERLINE RIGHT SOFT DIVIDER",
    0xE0B4: "POWERLINE RIGHT HALF CIRCLE THICK",
    0xE0B5: "POWERLINE RIGHT HALF CIRCLE THIN",
    0xE0B6: "POWERLINE LEFT HALF CIRCLE THICK",
    0xE0B7: "POWERLINE LEFT HALF CIRCLE THIN",
    0xE0B8: "POWERLINE LOWER LEFT TRIANGLE",
    0xE0B9: "POWERLINE BACKSLASH SEPARATOR",
    0xE0BA: "POWERLINE LOWER RIGHT TRIANGLE",
    0xE0BB: "POWERLINE FORWARDSLASH SEPARATOR",
    0xE0BC: "POWERLINE UPPER LEFT TRIANGLE",
    0xE0BD: "POWERLINE FORWARDSLASH SEPARATOR REDUNDANT",
    0xE0BE: "POWERLINE UPPER RIGHT TRIANGLE",
    0xE0BF: "POWERLINE BACKSLASH SEPARATOR REDUNDANT",
    0xE0C0: "POWERLINE FLAME THICK",
    0xE0C1: "POWERLINE FLAME THIN",
    0xE0C2: "POWERLINE FLAME THICK MIRRORED",
    0xE0C3: "POWERLINE FLAME THIN MIRRORED",
    0xE0C4: "POWERLINE PIXELATED SQUARES SMALL",
    0xE0C5: "POWERLINE PIXELATED SQUARES SMALL MIRRORED",
    0xE0C6: "POWERLINE PIXELATED SQUARES BIG",
    0xE0C7: "POWERLINE PIXELATED SQUARES BIG MIRRORED",
    0xE0C8: "POWERLINE ICE WAVEFORM",
    0xE0CA: "POWERLINE ICE WAVEFORM MIRRORED",
    0xE0CC: "POWERLINE HONEYCOMB",
    0xE0CD: "POWERLINE HONEYCOMB OUTLINE",
    0xE0CE: "POWERLINE LEGO SEPARATOR",
    0xE0CF: "POWERLINE LEGO SEPARATOR THIN",
    0xE0D0: "POWERLINE LEGO BLOCK FACING",
    0xE0D1: "POWERLINE LEGO BLOCK SIDEWAYS",
    0xE0D2: "POWERLINE TRAPEZOID TOP BOTTOM",
    0xE0D4: "POWERLINE TRAPEZOID TOP BOTTOM MIRRORED",
    0xE0D6: "POWERLINE RIGHT HARD DIVIDER INVERSE",
    0xE0D7: "POWERLINE LEFT HARD DIVIDER INVERSE",
}

# Sections in page order. Letter sections are added per script.
SECTIONS = (
    [
        f"{script_name} {kind}"
        for _, script_name in SCRIPTS
        for kind in ("uppercase", "lowercase", "small capitals", "other")
    ]
    + [
        "Numbers",
        "Punctuation",
        "Modifier letters",
        "Combining marks",
        # Symbols used in text
        "Currency",
        "Math",
        "Arrows",
        "Geometric shapes",
        "Technical",
        "Enclosed",
        "Symbols",
        # Text-mode graphics
        "Box drawing",
        "Block elements",
        "Sextants and octants",
        "Braille",
        "Legacy computing",
        "Powerline",
        # Nothing to see
        "Spaces and invisibles",
        "Other",
    ]
)

BLOCK_SECTIONS = {
    "Superscripts and Subscripts": "Numbers",
    "Number Forms": "Numbers",
    "Variation Selectors": "Spaces and invisibles",
    "Mathematical Operators": "Math",
    "Supplemental Mathematical Operators": "Math",
    "Miscellaneous Mathematical Symbols-A": "Math",
    "Miscellaneous Mathematical Symbols-B": "Math",
    "Mathematical Alphanumeric Symbols": "Math",
    "Arrows": "Arrows",
    "Supplemental Arrows-A": "Arrows",
    "Supplemental Arrows-B": "Arrows",
    "Supplemental Arrows-C": "Arrows",
    "Miscellaneous Technical": "Technical",
    "Control Pictures": "Technical",
    "Optical Character Recognition": "Technical",
    "Private Use Area": "Powerline",
    "Box Drawing": "Box drawing",
    "Block Elements": "Block elements",
    "Symbols for Legacy Computing": "Legacy computing",
    "Symbols for Legacy Computing Supplement": "Legacy computing",
    "Geometric Shapes": "Geometric shapes",
    "Geometric Shapes Extended": "Geometric shapes",
    "Miscellaneous Symbols and Arrows": "Geometric shapes",
    "Enclosed Alphanumerics": "Enclosed",
    "Enclosed Alphanumeric Supplement": "Enclosed",
    "Letterlike Symbols": "Symbols",
    "Miscellaneous Symbols": "Symbols",
    "Dingbats": "Symbols",
    "Miscellaneous Symbols and Pictographs": "Symbols",
    "Braille Patterns": "Braille",
}

# Name-derived variants sort after the accented ones
NAMED_VARIANT = 0x110000
DIGRAPH_VARIANT = 0x220000


def get_name(code):
    return POWERLINE_NAMES.get(code) or unicodedata.name(
        chr(code), f"U+{code:04X}"
    )


def get_base(char, script):
    """Returns the base letter and a sort key of the variant: the combining
    marks of the decomposition, or pseudo marks for variants found by name
    (O WITH STROKE, TURNED A, digraph AE)."""
    decomposition = unicodedata.normalize("NFD", char)
    if len(decomposition) > 1 and unicodedata.category(decomposition[0])[0] == "L":
        base, key = get_base(decomposition[0], script)
        return base, key + tuple(ord(mark) for mark in decomposition[1:])

    match = re.fullmatch(
        r"(\w+) (CAPITAL|SMALL) LETTER (.+?)(?: WITH .*)?",
        unicodedata.name(char, ""),
    )
    if not match:
        return char, ()
    prefix = f"{match[1]} {match[2]} LETTER "
    letter = match[3]
    candidates = [(letter, NAMED_VARIANT)]
    if script in ("Latn", "Grek"):
        # TURNED A, SCRIPT G, CLOSED OPEN E, FINAL SIGMA
        candidates.append((letter.split(" ")[-1], NAMED_VARIANT))
    if script == "Latn" and re.fullmatch(r"[A-Z]{2}", letter):
        candidates.append((letter[0], DIGRAPH_VARIANT))
    for candidate, variant in candidates:
        try:
            base = unicodedata.lookup(prefix + candidate)
        except KeyError:
            continue
        if base != char:
            base, key = get_base(base, script)
            return base, key + (variant + ord(char),)
    return char, ()


def get_section(code):
    char = chr(code)
    category = unicodedata.category(char)
    block = fontunicodedata.block(char)
    script = fontunicodedata.script(char)
    if script == "Copt":
        script = "Grek"
    name = get_name(code)
    script_name = dict(SCRIPTS).get(script)

    if category[0] == "Z" or category == "Cf" or block == "Variation Selectors":
        return "Spaces and invisibles"
    if category == "Sc":
        return "Currency"
    if category[0] == "M":
        return "Combining marks"
    if code < 0x80:
        if category[0] == "L":
            return f"Latin {'uppercase' if category == 'Lu' else 'lowercase'}"
        if category == "Nd":
            return "Numbers"
        if category == "Sm":
            return "Math"
        return "Punctuation"
    # Not LATIN CAPITAL LETTER SMALL CAPITAL I, the uppercase of ɪ
    if "SMALL CAPITAL" in name and category == "Ll" and script_name:
        return f"{script_name} small capitals"
    # Ↄ ↄ in Number Forms are letters
    if block in BLOCK_SECTIONS and not (
        block == "Number Forms" and category[0] == "L"
    ):
        section = BLOCK_SECTIONS[block]
        if section == "Legacy computing" and re.match(
            r"BLOCK (SEXTANT|OCTANT)", name
        ):
            return "Sextants and octants"
        # Arrows and arrowheads among the shapes and dingbats
        if section in ("Geometric shapes", "Symbols") and re.search(
            r"\bARROW", name
        ):
            return "Arrows"
        return section
    if script_name and script in ("Cyrl", "Armn") and category[0] != "L":
        return f"{script_name} other"
    if category in ("Lm", "Sk"):
        return "Modifier letters"
    if category[0] == "L":
        if not script_name:
            # MICRO SIGN
            return "Symbols"
        if category in ("Lu", "Lt"):
            return f"{script_name} uppercase"
        if category == "Ll":
            return f"{script_name} lowercase"
        return f"{script_name} other"
    if category[0] == "N":
        return "Numbers"
    if category[0] == "P":
        return "Punctuation"
    if category == "Sm":
        return "Math"
    if category == "So":
        return "Symbols"
    return "Other"


def get_sort_key(code, section):
    """Returns (group, key): a new group starts a run of related glyphs."""
    char = chr(code)
    category = unicodedata.category(char)
    name = get_name(code)
    if section.endswith(("uppercase", "lowercase", "small capitals")):
        script = fontunicodedata.script(char)
        if section.endswith("small capitals"):
            # Group by the letter: LATIN LETTER SMALL CAPITAL L WITH STROKE -> L
            letter = re.search(r"SMALL CAPITAL (?:LETTER )?(\S+)", name)[1]
            return (letter,), (" WITH " in name, code)
        base, key = get_base(char, script)
        base_code = ord(base)
        core = any(
            first <= base_code <= last
            for first, last in CORE_LETTERS.get(script, [])
        )
        # The base letter first, then its variants
        return (not core, base_code), (bool(key),) + key + (code,)
    if section == "Sextants and octants":
        return ("OCTANT" in name,), (code,)
    if section == "Numbers":
        if 0x30 <= code <= 0x39:
            return (0,), (code,)
        if "FRACTION" in name or "VULGAR" in name:
            return (2,), (code,)
        if "ROMAN NUMERAL" in name:
            return (3,), (code,)
        return (1,), (code,)
    if section == "Punctuation":
        if "QUOTATION" in name or "APOSTROPHE" in name or category in ("Pi", "Pf"):
            return (3,), (code,)
        if category in ("Ps", "Pe"):
            return (2,), (code,)
        if category == "Pd" or "HYPHEN" in name or "DASH" in name:
            return (1,), (code,)
        return (0,), (code,)
    return (), (code,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", help="the page to update")
    parser.add_argument(
        "--family",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "ROMAN", "ITALIC"),
        help="a family name with its roman and italic variable fonts",
    )
    args = parser.parse_args()

    html_dir = os.path.dirname(os.path.abspath(args.html))

    def get_url(path):
        relative = os.path.relpath(os.path.abspath(path), html_dir)
        return urllib.parse.quote(relative.replace(os.sep, "/"), safe="/,")

    families = []
    axes = {}
    cmaps = []
    for name, roman, italic in args.family:
        font = TTFont(roman)
        cmap = set(font.getBestCmap())
        cmaps.append(cmap)
        units_per_em = font["head"].unitsPerEm
        glyf = font["glyf"]
        advances = []
        top = font["hhea"].ascent
        bottom = font["hhea"].descent
        for glyph_name in font.getBestCmap().values():
            advances.append(font["hmtx"][glyph_name][0])
            glyph = glyf[glyph_name]
            if glyph.numberOfContours:
                glyph.recalcBounds(glyf)
                top = max(top, glyph.yMax)
                bottom = min(bottom, glyph.yMin)
        advances.sort()
        for axis in font["fvar"].axes:
            axes[axis.axisTag] = {
                "tag": axis.axisTag,
                "name": font["name"].getDebugName(axis.axisNameID),
                "min": axis.minValue,
                "default": axis.defaultValue,
                "max": axis.maxValue,
            }
        families.append(
            {
                "name": name,
                "roman": get_url(roman),
                "italic": get_url(italic),
                # Wide enough for 95% of the glyphs; the margin takes the rest
                "advance": round(
                    advances[int(len(advances) * 0.95)] / units_per_em, 4
                ),
                "widest": advances[-1] / units_per_em,
                # Line and ink extents, so stacked accents fit the cell
                "ascent": font["hhea"].ascent / units_per_em,
                "descent": -font["hhea"].descent / units_per_em,
                "top": top / units_per_em,
                "bottom": -bottom / units_per_em,
            }
        )
    axes["ital"] = {"tag": "ital", "name": "Italic", "min": 0, "default": 0, "max": 1}
    axes = [axes[tag] for tag in AXIS_ORDER if tag in axes]

    all_codes = sorted(set().union(*cmaps))
    sections = {section: [] for section in SECTIONS}
    for code in all_codes:
        if code < 0x20 or 0x7F <= code < 0xA0:
            continue
        section = get_section(code)
        sections[section].append(code)

    table = []
    for section, codes in sections.items():
        if not codes:
            continue
        keyed = sorted((get_sort_key(code, section), code) for code in codes)
        glyphs = []
        previous_group = None
        for (group, _), code in keyed:
            glyph = [code, get_name(code)]
            mask = sum(1 << i for i, cmap in enumerate(cmaps) if code in cmap)
            lead = int(group != previous_group and bool(group))
            previous_group = group
            if lead or mask != (1 << len(cmaps)) - 1:
                glyph.append(lead)
            if mask != (1 << len(cmaps)) - 1:
                glyph.append(mask)
            glyphs.append(glyph)
        table.append([section, glyphs])

    data = (
        "const FAMILIES = "
        + json.dumps(families, ensure_ascii=False)
        + ";\nconst AXES = "
        + json.dumps(axes, ensure_ascii=False)
        + ";\nconst SECTIONS = [\n"
        + ",\n".join(
            json.dumps(section, ensure_ascii=False, separators=(",", ":"))
            for section in table
        )
        + "\n];\n"
    )

    with open(args.html, encoding="utf-8", newline="") as file:
        html = file.read()
    newline = "\r\n" if "\r\n" in html else "\n"
    begin = html.index("// BEGIN GLYPHS")
    begin = html.index("\n", begin) + 1
    end = html.index("// END GLYPHS")
    end = html.rindex("\n", 0, end) + 1
    html = html[:begin] + data.replace("\n", newline) + html[end:]
    title = " · ".join(family["name"] for family in families)
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1)
    with open(args.html, "w", encoding="utf-8", newline="") as file:
        file.write(html)

    print(f"{args.html}: {sum(len(g) for _, g in table)} glyphs")


if __name__ == "__main__":
    main()
