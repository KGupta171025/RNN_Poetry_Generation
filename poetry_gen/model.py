from keras.models import Sequential, load_model as keras_load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
from poetry_gen.logger import logger
from poetry_gen.exceptions import ModelLoadError, ModelTrainingError

def build_model(seq_length, num_characters, learning_rate=0.01):
    """
    Creates and compiles a Sequential LSTM model for character prediction.
    """
    logger.info(f"Building Sequential LSTM model (input_shape=({seq_length}, {num_characters}))...")
    try:
        model = Sequential()
        model.add(LSTM(128, input_shape=(seq_length, num_characters)))
        model.add(Dense(num_characters))
        model.add(Activation('softmax'))
        
        logger.info(f"Compiling model with RMSprop optimizer (learning_rate={learning_rate})...")
        optimizer = RMSprop(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model
    except Exception as e:
        logger.error(f"Failed to build or compile the model: {e}")
        raise ModelTrainingError(f"Error building/compiling model: {e}") from e

def save_model(model, filepath):
    """
    Saves the trained model to filepath. Appends .keras if extension is invalid.
    """
    if not (filepath.endswith('.keras') or filepath.endswith('.h5')):
        logger.warning(f"Filepath '{filepath}' does not have a valid extension (.keras or .h5). Changing to '{filepath}.keras'")
        filepath += '.keras'
        
    logger.info(f"Saving Keras model to {filepath}...")
    try:
        model.save(filepath)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save the model to {filepath}: {e}")
        raise ModelTrainingError(f"Error saving model: {e}") from e

def load_model(filepath):
    """
    Loads a saved Keras model from filepath.
    """
    logger.info(f"Loading Keras model from {filepath}...")
    try:
        model = keras_load_model(filepath)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load the model from {filepath}: {e}")
        raise ModelLoadError(f"Error loading model from {filepath}: {e}") from e
