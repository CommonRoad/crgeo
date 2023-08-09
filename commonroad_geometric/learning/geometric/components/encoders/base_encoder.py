from abc import abstractmethod


class BaseEncoder:
    """Base class for all custom encoders.

        Your models must extend this class for the encoding decoding paradigm.
    """

    @abstractmethod
    def _build(self, num_features, hidden_channels, **kwargs) -> None:
        """Base method for instantiation that every encoder must extend.
        Args:
            num_features: The feature dimensions for the input
            hidden_channels: The dimensions for the hidden layer
        """
