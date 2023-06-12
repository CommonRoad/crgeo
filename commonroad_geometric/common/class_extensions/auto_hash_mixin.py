import uuid

class AutoHashMixin:
    """
    Automatically generates hash for instances of class.
    """

    def __hash__(self) -> int:
        if not hasattr(self, '__uuidhash__'):
            self.__uuidhash__ = int(uuid.uuid4())
        return self.__uuidhash__
