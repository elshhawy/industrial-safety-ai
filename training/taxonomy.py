"""Six-class PPE taxonomy from IMPLEMENTATION_SPEC.md.

Source class IDs are never assumed. Mapping is by source class name after
normalization. Aliases cover documented SH17 naming variants; they are only
applied when a source name matches after normalization.
"""

from __future__ import annotations

TARGET_CLASS_NAMES: list[str] = [
    "person",
    "Face-covering",
    "glasses",
    "gloves",
    "helmet",
    "suit",
]

TARGET_NAME_TO_ID: dict[str, int] = {
    name: index for index, name in enumerate(TARGET_CLASS_NAMES)
}

# Required source names from IMPLEMENTATION_SPEC.md Section 3.
SOURCE_TO_TARGET_NAME: dict[str, str] = {
    "person": "person",
    "face-guard": "Face-covering",
    "face-mask": "Face-covering",
    "glasses": "glasses",
    "gloves": "gloves",
    "helmet": "helmet",
    "medical-suit": "suit",
    "safety-suit": "suit",
    "safety-vest": "suit",
}

# Documented aliases. Keys must already be passed through normalize_class_name.
SOURCE_NAME_ALIASES: dict[str, str] = {
    "face-mask-medical": "face-mask",
    "facemask": "face-mask",
    "face-guard-medical": "face-guard",
    "earmuffs": "ear-mufs",
    "ear-muffs": "ear-mufs",
    "tools": "tool",
}

REQUIRED_SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "Face-covering": ("face-guard", "face-mask"),
    "glasses": ("glasses",),
    "gloves": ("gloves",),
    "helmet": ("helmet",),
    "suit": ("medical-suit", "safety-suit", "safety-vest"),
}


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def resolve_source_name(name: str) -> str:
    normalized = normalize_class_name(name)
    return SOURCE_NAME_ALIASES.get(normalized, normalized)


def class_mapping_summary() -> str:
    """Short parameter-safe summary. Full mapping belongs in a JSON artifact."""
    parts: list[str] = []
    for target_name in TARGET_CLASS_NAMES:
        sources = [
            source
            for source, mapped in SOURCE_TO_TARGET_NAME.items()
            if mapped == target_name
        ]
        target_id = TARGET_NAME_TO_ID[target_name]
        parts.append(f"{target_id}:{target_name}<-{','.join(sources)}")
    return "; ".join(parts)


def build_source_id_to_target_id(
    source_names: dict[int, str],
) -> tuple[dict[int, int], dict[str, object]]:
    """Map source class IDs to target IDs 0-5 using names, not assumed IDs."""
    source_id_to_target_id: dict[int, int] = {}
    resolved_by_source_id: dict[int, str] = {}
    present_canonical: set[str] = set()

    for source_id, raw_name in source_names.items():
        canonical = resolve_source_name(raw_name)
        resolved_by_source_id[int(source_id)] = canonical
        present_canonical.add(canonical)
        target_name = SOURCE_TO_TARGET_NAME.get(canonical)
        if target_name is None:
            continue
        source_id_to_target_id[int(source_id)] = TARGET_NAME_TO_ID[target_name]

    missing_targets: list[str] = []
    for target_name, source_group in REQUIRED_SOURCE_GROUPS.items():
        if not any(name in present_canonical for name in source_group):
            missing_targets.append(target_name)

    if missing_targets:
        available = [
            f"{source_id}:{name}" for source_id, name in sorted(source_names.items())
        ]
        raise ValueError(
            "Source dataset does not provide usable classes for target(s) "
            f"{missing_targets}. Available source classes: {available}. "
            "Inspect the downloaded SH17 metadata before changing the mapping."
        )

    if "person" not in present_canonical:
        raise ValueError(
            "Source dataset has no 'person' class. Person boxes will not be "
            "fabricated. Additional person annotations are required."
        )

    details: dict[str, object] = {
        "source_names": {str(k): v for k, v in sorted(source_names.items())},
        "resolved_source_names": {
            str(k): v for k, v in sorted(resolved_by_source_id.items())
        },
        "source_id_to_target_id": {
            str(k): v for k, v in sorted(source_id_to_target_id.items())
        },
        "target_class_names": list(TARGET_CLASS_NAMES),
        "present_canonical_source_names": sorted(present_canonical),
        "unmapped_source_names": sorted(
            name for name in present_canonical if name not in SOURCE_TO_TARGET_NAME
        ),
    }
    return source_id_to_target_id, details
