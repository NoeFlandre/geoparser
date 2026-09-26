"""Plain-text descriptions of gazetteer features.

Shared by :class:`SentenceTransformerResolver`, which embeds these
descriptions, and the annotator, which shows them to people choosing a
candidate. Keeping one copy keeps the two from drifting apart.
"""

import typing as t

# Gazetteer-specific attribute mappings for location descriptions
GAZETTEER_ATTRIBUTE_MAP: dict[str, dict[str, str]] = {
    "geonames": {
        "name": "name",
        "type": "feature_name",
        "level1": "country_name",
        "level2": "admin1_name",
        "level3": "admin2_name",
    },
    "geonames-cities": {
        "name": "name",
        "type": "feature_name",
        "level1": "country_name",
        "level2": "admin1_name",
        "level3": "admin2_name",
    },
    "swissnames3d": {
        "name": "NAME",
        "type": "OBJEKTART",
        "level1": "KANTON_NAME",
        "level2": "BEZIRK_NAME",
        "level3": "GEMEINDE_NAME",
    },
}


def admin_levels(data: t.Mapping[str, t.Any], attr_map: t.Mapping[str, str]) -> list:
    """
    Administrative place names for a feature, most specific first.

    Args:
        data: The feature's gazetteer attributes
        attr_map: Attribute mapping for the feature's gazetteer

    Returns:
        The non-empty administrative names, in level3..level1 order
    """
    values = []
    for level in ("level3", "level2", "level1"):
        if level in attr_map:
            value = data.get(attr_map[level])
            if value:
                values.append(value)
    return values


def describe_feature(data: t.Mapping[str, t.Any], attr_map: t.Mapping[str, str]) -> str:
    """
    Describe a feature as "name (type) in level3, level2, level1".

    Args:
        data: The feature's gazetteer attributes
        attr_map: Attribute mapping for the feature's gazetteer

    Returns:
        The description, or "" when none of the mapped attributes are set
    """
    description_parts = []

    feature_name = data.get(attr_map["name"])
    if feature_name:
        description_parts.append(feature_name)

    feature_type = data.get(attr_map["type"])
    if feature_type:
        description_parts.append(f"({feature_type})")

    levels = admin_levels(data, attr_map)
    if levels:
        description_parts.append("in")
        description_parts.append(", ".join(levels))

    return " ".join(description_parts).strip()
