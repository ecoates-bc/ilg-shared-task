import re
import pathlib


class EnglishStemmerDataset:
    def __init__(self, language: str):
        self.language = language

        if language == "ddo":
            lang_folder = "Tsez"
        elif language == "git":
            lang_folder = "Gitksan"
        elif language == "arp":
            lang_folder = "Arapaho"
        elif language == "ntu":
            lang_folder = "Natugu"
        elif language == "lez":
            lang_folder = "Lezgi"
        elif language == "nyb":
            lang_folder = "Nyangbo"
        elif language == "usp":
            lang_folder = "Uspanteko"
        else:
            raise ValueError("Bad language input.")

        self.train_orig_path = pathlib.Path("data") / lang_folder / f"{self.language}-train-track1-uncovered"
        self.dev_orig_path = pathlib.Path("data") / lang_folder / f"{self.language}-dev-track1-uncovered"
        self.test_orig_path = pathlib.Path("data") / lang_folder / f"{self.language}-test-track1-covered"

        self.train = self.get_train_entries()
        self.dev = self.get_dev_entries()
        self.test = self.get_test_entries()

    @staticmethod
    def _get_entry_dict(entry: str) -> dict:
        entry_as_list = entry.split("\n")
        return {
            "src": entry_as_list[0].replace("\\t ", ""),
            "gloss": entry_as_list[1].replace("\\g ", ""),
            "translation": entry_as_list[2].replace("\\l ", ""),
        }

    @staticmethod
    def _tokenize_src(translation_seq: str) -> str:
        lowered = translation_seq.lower()
        punc_replace = re.sub("[.,;:!-']|\"|\(\w+\)", "", lowered)
        
        return " ".join(list(punc_replace.replace(" ",  "_")))

    @staticmethod
    def _get_stem_sequence(gloss_seq: str) -> str:
        tokens = re.split("\s|[.-]|~", gloss_seq)
        only_stems = filter(lambda tok: re.match("[A-Z]?[a-z]+", tok), tokens)
        stem_seq = "_".join(list(only_stems))
        return " ".join(list(stem_seq)).lower()

    def _tokenize_entry(self, entry: dict) -> dict:
        src_tokens = self._tokenize_src(entry["translation"])
        stem_tokens = self._get_stem_sequence(entry["gloss"])
        return {"src": src_tokens, "stems": stem_tokens}

    def get_train_entries(self):
        train_text = self.train_orig_path.read_text().strip()
        train_entries = train_text.split("\n\n")
        train_entries = [self._get_entry_dict(entry) for entry in train_entries]

        return [self._tokenize_entry(e) for e in train_entries]

    def get_dev_entries(self):
        dev_text = self.dev_orig_path.read_text().strip()
        dev_entries = dev_text.split("\n\n")
        dev_entries = [self._get_entry_dict(entry) for entry in dev_entries]

        return [self._tokenize_entry(e) for e in dev_entries]

    def get_test_entries(self):
        test_text = self.test_orig_path.read_text().strip()
        test_entries = test_text.split("\n\n")
        test_entries = [self._get_entry_dict(entry) for entry in test_entries]

        return [self._tokenize_entry(e) for e in test_entries]
