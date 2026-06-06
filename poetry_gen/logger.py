import logging
import sys

def setup_logger(verbose=False):
    """
    Sets up a logger that outputs to stdout (console) and logs details to poetry_generation.log.
    """
    logger = logging.getLogger("poetry_gen")
    
    # Set to DEBUG if verbose is true, else INFO
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Avoid duplicate handler registration
    if logger.hasHandlers():
        # Update level of existing logger if called again
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger

    # Standard log format for file output
    file_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console format (clean and readable for users)
    console_format = logging.Formatter("%(levelname)s: %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler("poetry_generation.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger

# Get a package-wide logger module instance
logger = logging.getLogger("poetry_gen")
