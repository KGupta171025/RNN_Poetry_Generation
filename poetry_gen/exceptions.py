class PoetryGenError(Exception):
    """Base exception class for all poetry_gen errors."""
    pass

class DataDownloadError(PoetryGenError):
    """Raised when downloading or loading the training corpus fails."""
    pass

class ModelLoadError(PoetryGenError):
    """Raised when loading the Keras model fails."""
    pass

class ModelTrainingError(PoetryGenError):
    """Raised when training the LSTM model fails."""
    pass

class GenerationError(PoetryGenError):
    """Raised when text generation parameters are invalid or generation fails."""
    pass
