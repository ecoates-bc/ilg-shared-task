import pathlib
import re
import math


def get_nbest_predictions_for_word(row_number: int, output_text: str) -> list:
    nbest = {}
    sys_matches = re.findall(f"H-{row_number}\s+([-.\d]+)\s+([^\n]*)\n", output_text)
    for match in sys_matches:
        nbest.update({match[1]: float(match[0])})

    return nbest


def vote_on_best_matches(matches: list) -> list:
    all_predictions = set(matches[0].keys())
    for match in matches:
        all_predictions.update(match.keys())

    max_nll = -100
    best_token = ""
    for token in all_predictions:
        sum_nll = 0
        for predictions in matches:
            if token in predictions.keys():
                sum_nll += predictions[token]
        if sum_nll > max_nll:
            max_nll = sum_nll
            best_token = token

    return best_token


def get_sentence_level_predictions(output_texts: list, stem_output: str, dataset: list):
    output_sentences = []
    total_token_counter = 0
    stems = stem_predictions(stem_output)

    for entry in dataset:
        src_tokens = entry["src"]
        predicted_gloss = []
        for token in src_tokens:
            ensemble_matches = [get_nbest_predictions_for_word(total_token_counter, text) for text in output_texts]
            if ensemble_matches:
                best_match = vote_on_best_matches(ensemble_matches)
                predicted_gloss.append(best_match.replace(" ", ""))
            total_token_counter += 1
        sentence_stems = next(stems)
        output_gloss = create_sentence_level_gloss(predicted_gloss, sentence_stems)
        output_sentences.append(output_gloss)

    return output_sentences


def get_predicted_stems(sentence_number: int, stem_output_text: str) -> list:
    sys_match = re.search(f"H-{sentence_number}\s+[-.\d]+\s+([^\n]*)\n", stem_output_text)
    if sys_match:
        raw_stems = sys_match.group(1).replace(" ", "")
        tokens = raw_stems.split("_")
        return tokens

def create_sentence_level_gloss(gloss_tokens: list, predicted_stems: list) -> str:
    n_tokens = len(gloss_tokens)
    n_stems = len(predicted_stems)
    output = []

    stem_counter = 0
    for i in range(n_tokens):
        if "[STEM]" in gloss_tokens[i]:
            if stem_counter < n_stems:
                replaced = gloss_tokens[i].replace("[STEM]", predicted_stems[stem_counter])
                output.append(replaced)
                stem_counter += 1
            else:
                output.append(gloss_tokens[i])
        else:
            output.append(gloss_tokens[i])

    print(gloss_tokens)
    print(predicted_stems)
    print(output)
    print()

    return " ".join(output)


def stem_predictions(stem_output_text: str) -> iter:
    stem_row = 0
    stems = get_predicted_stems(stem_row, stem_output_text)
    while stems:
        yield stems
        stem_row += 1
        stems = get_predicted_stems(stem_row, stem_output_text)


def save_predictions_file(output_text: list, stem_output: str, dataset: list, covered_path: pathlib.Path, saved_path: pathlib.Path):
    sentence_predictions = get_sentence_level_predictions(output_text, stem_output, dataset)

    covered_text = covered_path.read_text().strip()
    covered_entries = covered_text.split("\n\n")

    prediction_entries = []
    for i in range(len(covered_entries)):
        entry = covered_entries[i]
        entry = entry.replace("\\g ", f"\\g {sentence_predictions[i]}")
        prediction_entries.append(entry)

    saved_path.write_text("\n\n".join(prediction_entries))
