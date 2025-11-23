from typing import TypeVar
from pydantic.fields import FieldInfo

T = TypeVar("T")


def extract_unique_metadata(
    field: FieldInfo,
    type_: type[T],
    field_name: str,
    exception_class: type[BaseException],
) -> T:
    """
    Extracts a unique metadata object of type T from the FieldInfo object provided.
    Raises an exception if the field is not annotated with the specified type or if multiple
    metadata objects of the specified type are found.
    """
    metadata = field.metadata or ()
    matches = [m for m in metadata if isinstance(m, type_)]
    if not matches:
        raise exception_class(
            f"Field '{field_name}' not annotated with {type_.__name__}."
        )
    if len(matches) > 1:
        raise exception_class(
            f"Field '{field_name}' annotated with multiple {type_.__name__} objects."
        )
    return matches[0]


__all__ = ["extract_unique_metadata"]
