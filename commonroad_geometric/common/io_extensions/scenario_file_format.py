from enum import Enum, auto


class ScenarioFileFormat(Enum):
    """
    This enum is used to indicate which file format to use for saving and loading scenarios.
    XML indicates to read and write scenario files with the CommonRoadFileReader and CommonRoadFileWriter.
    BUNDLE indicates to save or load a ScenarioBundle with pickle.
    ALL includes XML and BUNDLE.
    """
    XML = auto(), ['.xml']
    # PROTOBUF = FileFormat.PROTOBUF, ['.proto']  # Protobuf format supported
    BUNDLE = auto(), ['.pkl']
    ALL = auto(), ['.pkl', '.xml']

    def __init__(
        self,
        identifier: int,
        suffixes: list[str],
    ):
        self.identifier = identifier
        self.suffixes = suffixes
