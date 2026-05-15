# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


from dataclasses import fields
from typing import Any, List, Mapping

from fairseq2.typing import DataClass, is_dataclass_instance


def update_dataclass(
    obj: DataClass,
    overrides: Mapping[str, Any],
) -> List[str]:
    """Update ``obj`` with the data contained in ``overrides`` Return the unknown fields.
    Copied from an old version of fairseq2 with simplification.

    :param obj:
        The data class instance to update.
    :param overrides:
        The dictionary containing the data to set in ``obj``.
    """

    unknown_fields: List[str] = []

    field_path: List[str] = []

    # The dataset config has a special attribute `silent_freeze` that does not allow hard update
    forbidden_fields_ = ["silent_freeze"]

    def update(obj_: DataClass, overrides_: Mapping[str, Any]) -> None:
        overrides_copy = {**overrides_}

        for field in fields(obj_):
            if field.name in forbidden_fields_:
                continue
            value = getattr(obj_, field.name)

            try:
                override = overrides_copy.pop(field.name)
            except KeyError:
                continue

            # Recursively traverse child dataclasses.
            if override is not None and is_dataclass_instance(value):
                if not isinstance(override, Mapping):
                    pathname = ".".join(field_path + [field.name])

                    raise RuntimeError(
                        pathname,
                        f"The field '{pathname}' is expected to be of type `{type(value)}`, but is of type `{type(override)}` instead.",  # fmt: skip
                    )

                field_path.append(field.name)

                update(value, override)

                field_path.pop()
            else:
                setattr(obj_, field.name, override)

        if overrides_copy:
            unknown_fields.extend(
                ".".join(field_path + [name]) for name in overrides_copy
            )

    update(obj, overrides)

    unknown_fields.sort()

    return unknown_fields
