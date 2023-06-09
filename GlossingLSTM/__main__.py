import pathlib
import subprocess
import re
import shutil
import torch

from GlossingLSTM import (
    WindowedWordDataset,
    EnglishStemmerDataset,
    save_dataset_files,
    save_stemmer_dataset_files,
    save_predictions_file,
)


def train_window_model(lang: str, window_size: int, batch_size=64, patience=2, lr=0.003, nbest=5):
    dataset = WindowedWordDataset(lang, window_size)
    data_folder = save_dataset_files(lang, window_size, dataset.train, dataset.dev, dataset.test)

    preprocess_folder = pathlib.Path(f"{lang}-w{window_size}-preprocessed")

    preprocess_args = [
        "fairseq-preprocess",
        "--source-lang", "src",
        "--target-lang", "gloss",
        "--trainpref", f"{data_folder}/{lang}-w{window_size}-train",
        "--validpref", f"{data_folder}/{lang}-w{window_size}-dev",
        "--destdir", f"{preprocess_folder}",
    ]

    print(f"Preprocessing model with window={window_size}...")
    subprocess.run(preprocess_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("="*100)

    checkpoint_dir = pathlib.Path("checkpoints")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    train_args = [
        "fairseq-train",
        f"{preprocess_folder}",
        "--arch", "lstm",
        "--optimizer", "adam",
        "--lr", f"{lr}",
        "--seed", f"{window_size}",
        "--lr-scheduler", "inverse_sqrt",
        "--batch-size", f"{batch_size}",
        "--patience", f"{patience}",
    ]

    print(f"Training model with window={window_size}...")
    subprocess.run(train_args, stdout=subprocess.PIPE)
    print("="*250)

    interactive_args = [
        "fairseq-interactive",
        "--path", "checkpoints/checkpoint_best.pt",
        "--input", f"{data_folder}/{lang}-w{window_size}-dev.src",
        "--nbest", f"{nbest}",
        f"{preprocess_folder}"
    ]

    result_path = pathlib.Path(f"results-window{window_size}.txt")
    with open(result_path, "w") as results:
        gen_proc = subprocess.run(interactive_args, stdout=results)

    return result_path.read_text(), dataset


def train_stemmer(lang: str, batch_size=64, patience=2, lr=0.003):
    stemmer_ds = EnglishStemmerDataset(lang)
    data_folder = save_stemmer_dataset_files(lang, stemmer_ds.train, stemmer_ds.dev, stemmer_ds.test)

    preprocess_folder = pathlib.Path(f"{lang}-stemmer-preprocessed")

    preprocess_args = [
        "fairseq-preprocess",
        "--source-lang", "src",
        "--target-lang", "stems",
        "--trainpref", f"{data_folder}/{lang}-stemmer-train",
        "--validpref", f"{data_folder}/{lang}-stemmer-dev",
        "--destdir", f"{preprocess_folder}",
    ]

    print(f"Preprocessing stemmer model...")
    subprocess.run(preprocess_args, stdout=subprocess.PIPE)
    print("="*100)

    checkpoint_dir = pathlib.Path("checkpoints")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    train_args = [
        "fairseq-train",
        f"{preprocess_folder}",
        "--arch", "lstm",
        "--optimizer", "adam",
        "--lr", f"{lr}",
        "--lr-scheduler", "inverse_sqrt",
        "--batch-size", f"{batch_size}",
        "--patience", f"{patience}",
    ]

    print(f"Training stemmer model...")
    subprocess.run(train_args, stdout=subprocess.PIPE)
    print("="*250)

    interactive_args = [
        "fairseq-interactive",
        "--path", "checkpoints/checkpoint_best.pt",
        "--input", f"{data_folder}/{lang}-stemmer-dev.src",
        f"{preprocess_folder}"
    ]

    result_path = pathlib.Path(f"results-stemmer.txt")
    with open(result_path, "w") as results:
        gen_proc = subprocess.run(interactive_args, stdout=results)

    return result_path.read_text(), stemmer_ds


if __name__ == "__main__":
    LANG = "ddo"
    BATCH_SIZE = 128
    PATIENCE = 2
    LR = 0.003

    print("CUDA status:", torch.cuda.is_available())

    window1_results, dataset = train_window_model(LANG, 1)
    window2_results, _ = train_window_model(LANG, 2)
    model_results = [window1_results, window2_results]

    # stemmer_results, _ = train_stemmer(LANG)

    saved_path = pathlib.Path("results.txt")
    save_predictions_file(model_results, "", dataset.dev, dataset.get_dev_covered_path(), saved_path)

    eval_args = [
        "python3",
        "baseline/src/eval.py",
        "--pred", f"{saved_path}",
        "--gold", f"{dataset.get_dev_uncovered_path()}"
    ]
    eval = subprocess.run(eval_args, capture_output=True)
    print(str(eval.stdout))
    print(str(eval.stderr))