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


class JobError(PlatformError):
    """A background video job cannot be started or moved on."""


class JobBusyError(JobError):
    """Another video job is planning or producing; one runs at a time."""
