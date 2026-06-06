import argparse
import sys
import os
import traceback

from poetry_gen.logger import setup_logger, logger
from poetry_gen.exceptions import PoetryGenError
from poetry_gen.dataset import Dataset
from poetry_gen.model import build_model, save_model, load_model
from poetry_gen.generator import generate_text

def run_train(args):
    dataset = Dataset(filepath=args.corpus_path, url=args.corpus_url)
    dataset.load_data()
    x, y = dataset.prepare_data(seq_length=args.seq_length, step_size=args.step_size)
    
    model = build_model(
        seq_length=args.seq_length, 
        num_characters=len(dataset.characters), 
        learning_rate=args.lr
    )
    
    logger.info(f"Starting model training for {args.epochs} epochs with batch size {args.batch_size}...")
    model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs)
    
    save_model(model, args.model_path)
    logger.info("Training process completed successfully.")

def run_generate(args):
    dataset = Dataset(filepath=args.corpus_path, url=args.corpus_url)
    dataset.load_data()
    
    model = load_model(args.model_path)
    
    # Run the text generation
    generated_poetry = generate_text(
        model=model,
        dataset=dataset,
        length=args.length,
        temperature=args.temperature,
        seed=args.seed,
        seq_length=args.seq_length
    )
    
    print("\n" + "="*40)
    print(f"GENERATED TEXT (Temp={args.temperature}):")
    print("="*40)
    print(generated_poetry)
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="RNN Poetry Generation CLI - Train LSTM models or generate Shakespeare-like poetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a new LSTM poetry model.")
    train_parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for RMSprop optimizer.")
    train_parser.add_argument("--model-path", type=str, default="textgenerator.keras", help="Path to save the trained model.")
    train_parser.add_argument("--corpus-path", type=str, default=None, help="Local path to text corpus (downloads if not specified).")
    train_parser.add_argument("--corpus-url", type=str, default="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt", help="URL to download corpus from.")
    train_parser.add_argument("--seq-length", type=int, default=40, help="Input sequence length in characters.")
    train_parser.add_argument("--step-size", type=int, default=3, help="Step size for sequence sliding window.")
    
    # Generate parser
    gen_parser = subparsers.add_parser("generate", help="Generate poetry from a trained LSTM model.")
    gen_parser.add_argument("--model-path", type=str, default="textgenerator.keras", help="Path to the trained model file (.keras or .h5).")
    gen_parser.add_argument("--length", type=int, default=300, help="Number of characters to generate.")
    gen_parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (lower is more deterministic, higher is more creative).")
    gen_parser.add_argument("--seed", type=str, default=None, help="Seed text to initialize generation (at least 40 chars recommended).")
    gen_parser.add_argument("--corpus-path", type=str, default=None, help="Local path to text corpus (used to initialize characters and fallback seeds).")
    gen_parser.add_argument("--corpus-url", type=str, default="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt", help="URL to download corpus from.")
    gen_parser.add_argument("--seq-length", type=int, default=40, help="Sequence length used when model was trained.")

    args = parser.parse_args()
    
    # Configure logging
    setup_logger(verbose=args.verbose)
    
    try:
        if args.command == "train":
            run_train(args)
        elif args.command == "generate":
            run_generate(args)
    except PoetryGenError as e:
        logger.error(f"Application error: {e}")
        # Print a user-friendly error to console
        print(f"\n[ERROR] {e}\nSee 'poetry_generation.log' for details.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled system error: {e}\n{traceback.format_exc()}")
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}\nSee 'poetry_generation.log' for full details.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()