import os
import numpy as np
from keras.utils import get_file
from poetry_gen.logger import logger
from poetry_gen.exceptions import DataDownloadError

class Dataset:
    def __init__(self, filepath=None, url="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"):
        self.url = url
        self.filepath = filepath
        self.text = ""
        self.characters = []
        self.char_to_index = {}
        self.index_to_char = {}

    def load_data(self):
        """Downloads/loads the corpus text, processes unique characters and mappings."""
        try:
            if not self.filepath:
                logger.info(f"Downloading corpus from {self.url}...")
                # Download using keras.utils.get_file, saved under cache
                self.filepath = get_file(
                    'shakespeare.txt',
                    self.url
                )
            
            logger.info(f"Reading corpus file from {self.filepath}...")
            with open(self.filepath, 'rb') as f:
                self.text = f.read().decode(encoding='utf-8').lower()
                
            # For fast execution, select a slice of text as originally defined
            if len(self.text) > 800000:
                self.text = self.text[300000:800000]
            elif len(self.text) > 500000:
                self.text = self.text[300000:]
                
            self.characters = sorted(list(set(self.text)))
            self.char_to_index = {c: i for i, c in enumerate(self.characters)}
            self.index_to_char = {i: c for i, c in enumerate(self.characters)}
            
            logger.info(f"Corpus loaded: {len(self.text)} characters. Unique characters: {len(self.characters)}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise DataDownloadError(f"Failed to download/load the corpus data: {e}") from e

    def prepare_data(self, seq_length=40, step_size=3):
        """Generates sequences and corresponding next character labels, then returns OHE arrays."""
        if not self.text:
            raise DataDownloadError("Corpus text has not been loaded. Call load_data() first.")
            
        logger.info(f"Preparing sequences with sequence_length={seq_length}, step_size={step_size}...")
        sentences = []
        next_characters = []
        
        for i in range(0, len(self.text) - seq_length, step_size):
            sentences.append(self.text[i:i + seq_length])
            next_characters.append(self.text[i + seq_length])
            
        num_sentences = len(sentences)
        num_chars = len(self.characters)
        logger.info(f"Extracted {num_sentences} training sequences.")
        
        # Use Python's built-in bool type since np.bool is deprecated in NumPy 2.x
        x = np.zeros((num_sentences, seq_length, num_chars), dtype=bool)
        y = np.zeros((num_sentences, num_chars), dtype=bool)
        
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                if char in self.char_to_index:
                    x[i, t, self.char_to_index[char]] = True
            next_char = next_characters[i]
            if next_char in self.char_to_index:
                y[i, self.char_to_index[next_char]] = True
                
        return x, y
