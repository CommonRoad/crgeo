import uuid
from abc import abstractmethod


class BaseGeom:
    r"""
    This class wraps a geometric object, to enable rendering it by different rendering frameworks.
    """

    def __init__(
        self,
        creator: str = "Unknown",
    ):
        r"""

        Args:
            creator (str): Creator of the Geom, e.g. usually __name__ of the render plugin.
        """
        self._creator = creator
        self._uuid = str(uuid.uuid4())

    @property
    def creator(self):
        r"""
        Returns:
            The creator of this geometry, e.g. usually a render plugin.
        """
        return self._creator

    @property
    def uuid(self):
        r"""
        Returns:
            The UUID of this geometry
        """
        return self._uuid

    def __repr__(self):
        r"""
        Returns:
            Info about this geometry: Render framework, object type, creator and UUID.
        """
        # Print without "self" prefix
        creator, uuid = self.creator, self.uuid
        return f"{type(self).__name__}({creator=}, {uuid=})"
