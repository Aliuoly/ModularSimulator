from pydantic.fields import FieldInfo

def extract_unique_metadata(field: FieldInfo, type_: type, field_name: str, exception_class: type[BaseException]):
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