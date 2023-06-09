import pathlib


def flatten_src_sentences(src_elems: list):
    flat_entries = []
    for entry in src_elems:
        for token in entry:
            flat_entries.append(" _ ".join(token))
    return flat_entries


def flatten_gloss_sentences(gloss_elems: list):
    flat_glosses = []
    for entry in gloss_elems:
        for token in entry:
            flat_glosses.append(token)
    return flat_glosses


def save_dataset_files(language: str, window: int, train_ds: list, dev_ds: list, test_ds: list):
    train_sources = [elem["src"] for elem in train_ds]
    train_sources = flatten_src_sentences(train_sources)
    train_glosses = [elem["gloss"] for elem in train_ds]
    train_glosses = flatten_gloss_sentences(train_glosses)

    dev_sources = [elem["src"] for elem in dev_ds]
    dev_sources = flatten_src_sentences(dev_sources)
    dev_glosses = [elem["gloss"] for elem in dev_ds]
    dev_glosses = flatten_gloss_sentences(dev_glosses)

    test_sources = [elem["src"] for elem in test_ds]
    test_sources = flatten_src_sentences(test_sources)

    data_folder = pathlib.Path(f"{language}-w{window}-fairseq-data")
    data_folder.mkdir(exist_ok=True)

    train_src_path = data_folder / f"{language}-w{window}-train.src"
    train_src_path.write_text("\n".join(train_sources))

    train_gloss_path = data_folder / f"{language}-w{window}-train.gloss"
    train_gloss_path.write_text("\n".join(train_glosses))

    dev_src_path = data_folder / f"{language}-w{window}-dev.src"
    dev_src_path.write_text("\n".join(dev_sources))

    dev_gloss_path = data_folder / f"{language}-w{window}-dev.gloss"
    dev_gloss_path.write_text("\n".join(dev_glosses))

    test_src_path = data_folder / f"{language}-w{window}-test.src"
    test_src_path.write_text("\n".join(test_sources))

    return data_folder


def save_stemmer_dataset_files(language: str, train_ds: list, dev_ds: list, test_ds: list):
    train_sources = [elem["src"] for elem in train_ds]
    train_stems = [elem["stems"] for elem in train_ds]

    dev_sources = [elem["src"] for elem in dev_ds]
    dev_stems = [elem["stems"] for elem in dev_ds]

    test_sources = [elem["src"] for elem in test_ds]

    data_folder = pathlib.Path(f"{language}-stemmer-fairseq-data")
    data_folder.mkdir(exist_ok=True)

    train_src_path = data_folder / f"{language}-stemmer-train.src"
    train_src_path.write_text("\n".join(train_sources))

    train_stems_path = data_folder / f"{language}-stemmer-train.stems"
    train_stems_path.write_text("\n".join(train_stems))

    dev_src_path = data_folder / f"{language}-stemmer-dev.src"
    dev_src_path.write_text("\n".join(dev_sources))

    dev_stems_path = data_folder / f"{language}-stemmer-dev.stems"
    dev_stems_path.write_text("\n".join(dev_stems))

    test_stems_path = data_folder / f"{language}-stemmer-test.src"
    test_stems_path.write_text("\n".join(test_sources))

    return data_folder