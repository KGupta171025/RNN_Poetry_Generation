import random
import numpy as np
from poetry_gen.logger import logger
from poetry_gen.exceptions import GenerationError

def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array (softmax output).
    """
    try:
        preds = np.asarray(preds).astype('float64')
        # Avoid division by zero and log of zero
        if temperature <= 0:
            return np.argmax(preds)
            
        preds = np.log(preds + 1e-10) / temperature
        exp_pred = np.exp(preds)
        preds = exp_pred / np.sum(exp_pred)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except Exception as e:
        logger.error(f"Sampling error: {e}")
        raise GenerationError(f"Error sampling from predictions: {e}") from e

def generate_text(model, dataset, length, temperature, seed=None, seq_length=40):
    """
    Generates text of specified length using the model, dataset, and temperature.
    If seed is not provided, extracts a random seed from the dataset text.
    """
    logger.info(f"Generating {length} characters of text with temperature={temperature}...")
    try:
        if not dataset.text:
            raise GenerationError("Dataset text corpus is not loaded.")
            
        # Determine the seed sentence
        if seed:
            if len(seed) < seq_length:
                logger.warning(f"Seed text is shorter than sequence length ({seq_length}). Padding seed with random corpus text.")
                start_index = random.randint(0, len(dataset.text) - seq_length + len(seed) - 1)
                seed = dataset.text[start_index:start_index + (seq_length - len(seed))] + seed
            elif len(seed) > seq_length:
                logger.info(f"Truncating seed text to the last {seq_length} characters.")
                seed = seed[-seq_length:]
            sentence = seed.lower()
        else:
            start_index = random.randint(0, len(dataset.text) - seq_length - 1)
            sentence = dataset.text[start_index:start_index + seq_length]
            
        logger.debug(f"Using seed: {repr(sentence)}")
        
        generated = sentence
        num_chars = len(dataset.characters)
        
        for i in range(length):
            x = np.zeros((1, seq_length, num_chars))
            for t, character in enumerate(sentence):
                if character in dataset.char_to_index:
                    x[0, t, dataset.char_to_index[character]] = 1.0
            
            predictions = model.predict(x, verbose=0)[0]
            next_index = sample(predictions, temperature)
            next_character = dataset.index_to_char[next_index]
            
            generated += next_character
            sentence = sentence[1:] + next_character
            
        return generated
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        raise GenerationError(f"Error during text generation: {e}") from e
