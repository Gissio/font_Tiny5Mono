"""
bdf2ufo

Ligature building for BDF to UFO font conversion.

(C) 2024-2026 Gissio
License: MIT
"""

import logging

import numpy as np

from .bdf_font import BDFFont
from .data import LIGATURE_SPACER_NAME, LIGATURE_SUFFIX
from .utils import Vec2

# Definitions
logger = logging.getLogger(__name__)


def build_ligatures(bdf_font: BDFFont) -> dict:
    """Add the ligature glyphs of the font and return their substitutions.

    A ligature glyph is an unencoded glyph named after the glyph names of its
    characters, joined by underscores, with a ".liga" suffix: for instance,
    "hyphen_greater.liga" for "->". It is drawn as wide as its characters
    together.

    In a monospace font, a ligature of n characters is drawn n cells wide. It
    is turned into a glyph one cell wide, whose drawing extends over the n - 1
    cells to its left, which an empty spacer glyph fills. So every glyph keeps
    the monospace advance, and the cursor still moves one cell at a time.

    The glyphs are added after all other glyphs, so that the jitter of the
    other glyphs does not depend on whether ligatures are enabled.

    Args:
        bdf_font: The BDF font object.

    Returns:
        Dictionary with "ligatures", a list of (ligature glyph name, list of
        component glyph names), longest first, and "spacer", the name of the
        spacer glyph of a monospace font, or None.
    """
    monospace = bdf_font.monospace

    ligatures = []
    for ligature_name, glyph in bdf_font.ligature_glyphs.items():
        components = _get_components(bdf_font, ligature_name)
        if components is None:
            continue

        if monospace:
            expected_advance = len(components) * monospace
            if glyph["advance"] != expected_advance:
                logger.warning(
                    "Ligature '%s' has advance %d, expected %d for %d monospace cells, skipping",
                    ligature_name,
                    glyph["advance"],
                    expected_advance,
                    len(components),
                )
                continue

            glyph = dict(glyph)
            glyph["offset"] = glyph["offset"] - Vec2(expected_advance - monospace, 0)
            glyph["advance"] = monospace

        bdf_font.glyphs[ligature_name] = glyph
        ligatures.append((ligature_name, components))

    spacer = None
    if ligatures and monospace:
        if LIGATURE_SPACER_NAME in bdf_font.glyphs:
            raise ValueError(
                f"The font has a glyph named '{LIGATURE_SPACER_NAME}', which is "
                "reserved for the ligature spacer"
            )

        spacer = LIGATURE_SPACER_NAME
        bdf_font.glyphs[spacer] = {
            "character": None,
            "bitmap": np.zeros((1, 1), np.uint8),
            "offset": Vec2(0),
            "advance": monospace,
        }

    # Longer ligatures first, so that "-->" is tried before "->"
    ligatures.sort(key=lambda ligature: -len(ligature[1]))

    logger.info("Added %d ligatures", len(ligatures))

    return {"ligatures": ligatures, "spacer": spacer}


def _get_components(bdf_font: BDFFont, ligature_name: str) -> list:
    """Return the glyph names of the characters of a ligature.

    Args:
        bdf_font: The BDF font object.
        ligature_name: The name of the ligature glyph, such as
            "hyphen_greater.liga".

    Returns:
        A list of glyph names, or None if the name does not describe a
        ligature of at least two encoded glyphs of the font.
    """
    component_names = ligature_name.removesuffix(LIGATURE_SUFFIX).split("_")

    for component_name in component_names:
        if (
            component_name not in bdf_font.glyphs
            or bdf_font.glyphs[component_name]["character"] is None
        ):
            logger.warning(
                "Ligature '%s' names '%s', which is not an encoded glyph of the font, skipping",
                ligature_name,
                component_name,
            )
            return None

    if len(component_names) < 2:
        logger.warning(
            "Ligature '%s' has fewer than two characters, skipping", ligature_name
        )
        return None

    return component_names


def get_ligature_feature(ligatures: dict) -> list:
    """Return the lines of the calt feature that substitutes the ligatures.

    A ligature is only substituted where its characters are not preceded or
    followed by another character of any ligature, that is, where the run of
    such characters is the ligature itself. So "====" or "<==>" are not
    partly substituted.

    Args:
        ligatures: The ligatures, from build_ligatures().

    Returns:
        A list of feature file lines, empty if the font has no ligatures.
    """
    ligature_list = ligatures.get("ligatures", [])
    spacer = ligatures.get("spacer")
    if len(ligature_list) == 0:
        return []

    component_names = []
    for _ligature_name, components in ligature_list:
        for component_name in components:
            if component_name not in component_names:
                component_names.append(component_name)

    lines = []
    lines.append(f"@ligature_components = [{' '.join(component_names)}];")
    lines.append("")

    # Without a spacer, each ligature substitutes from a lookup of its own,
    # referenced from the context
    if spacer is None:
        for ligature_name, components in ligature_list:
            lookup_name = _get_lookup_name(ligature_name) + "_sub"
            lines.append(f"lookup {lookup_name} {{")
            lines.append(f"    sub {' '.join(components)} by {ligature_name};")
            lines.append(f"}} {lookup_name};")
            lines.append("")

    lines.append("feature calt {")
    for ligature_name, components in ligature_list:
        lookup_name = _get_lookup_name(ligature_name)
        first, rest = components[0], " ".join(components[1:])

        lines.append(f"    lookup {lookup_name} {{")
        lines.append(f"        ignore sub @ligature_components {first}' {rest};")
        lines.append(f"        ignore sub {first}' {rest} @ligature_components;")

        if spacer is None:
            lines.append(
                f"        sub {first}' lookup {lookup_name}_sub "
                + " ".join(f"{name}'" for name in components[1:])
                + ";"
            )
        else:
            # Each character becomes the spacer, and the last one the
            # ligature. The rule for the last character comes first, as the
            # earlier characters are already spacers when it applies.
            for index in reversed(range(len(components))):
                context = [spacer] * index
                context.append(components[index] + "'")
                context += components[index + 1 :]
                target = ligature_name if index == len(components) - 1 else spacer
                lines.append(f"        sub {' '.join(context)} by {target};")

        lines.append(f"    }} {lookup_name};")
    lines.append("} calt;")
    lines.append("")

    return lines


def _get_lookup_name(ligature_name: str) -> str:
    """Return the name of the calt lookup of a ligature."""
    return "calt_" + ligature_name.removesuffix(LIGATURE_SUFFIX)
