class PlatformError(Exception):
    """Base class for all application-raised errors."""


class ConfigError(PlatformError):
    """Bad or missing configuration."""


class ModelProviderError(PlatformError):
    """A model provider call failed (network, auth, or bad response)."""


class ModelOutputParseError(ModelProviderError):
    """The model returned text that could not be parsed into the expected shape."""


class IngestionError(PlatformError):
    """Problem with the property input package (missing files, bad JSON, etc.)."""


class RenderingError(PlatformError):
    """HTML/PDF rendering failed."""


class WorkflowError(PlatformError):
    """A workflow node failed in a non-recoverable way."""
