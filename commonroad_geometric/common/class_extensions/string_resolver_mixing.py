import re
from typing import List, Sequence, Type, Union

class StringResolverMixin:
    """
    Resolves subclass for base class.
    """

    @classmethod
    def resolve(cls, value: str) -> Type: # TODO: how to refer to base class?
        """
        Resolves the provided string to a subtype of this type.

        Raises:
            KeyError: If string could not be resolved.

        Returns:
            _type_: Subclass whose name matches the specified value. 
        """
        subclasses = cls.__subclasses__()
        lookup_dict = {subcls.__name__.lower(): subcls for subcls in subclasses}
        value_lower = value.lower()
        if value_lower not in lookup_dict:
            raise KeyError(
                f"Unable to resolve '{value}' for {cls.__name__}: Valid subclasses are {[subcls.__name__ for subcls in subclasses]}."
            )
        return lookup_dict[value_lower]

    @classmethod
    def resolve_many(cls, values: Union[str, Sequence[str]]) -> List[Type]:
        """
        Resolves the provided list of strings to subtypes of this type.

        Raises:
            KeyError: If a string could not be resolved.

        Returns:
            List[_type_]: Subclasses whose name matches the specified value. 
        """

        if isinstance(values, str):
            values = re.split('\W+', values) # split by non-alpha characters

        return [cls.resolve(value) for value in values]
