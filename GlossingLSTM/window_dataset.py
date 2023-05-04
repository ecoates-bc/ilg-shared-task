import pathlib
import re


class WindowedWordDataset:
    def __init__(self, language: str, window_size: int):
        self.language = language
        self.window_size = window_size

        if language == "ddo":
            lang_folder = "Tsez"
        elif language == "git":
            lang_folder = "Gitksan"
        elif language == "arp":
            lang_folder = "Arapaho"
        elif language == "ntu":
            lang_folder = "Natugu"
        else:
            raise ValueError("Bad language input.")

        self.train_orig_path = pathlib.Path("data") / lang_folder / f"{self.language}-train-track1-uncovered"
        self.dev_orig_path = pathlib.Path("data") / lang_folder / f"{self.language}-dev-track1-uncovered"

        self.train = self.get_train_entries()
        self.dev = self.get_dev_entries()

    @staticmethod
    def _get_entry_dict(entry: str) -> dict:
        entry_as_list = entry.split("\n")
        return {
            "src": entry_as_list[0].replace("\\t ", ""),
            "gloss": entry_as_list[1].replace("\\g ", ""),
        }

    @staticmethod
    def _tokenize_src(src_seq: str) -> list:
        lowered = src_seq.lower()
        punc_replace = re.sub("[.,;:]", "", lowered)
        words = punc_replace.split(" ")
        return [' '.join(list(w)) for w in words]

    @staticmethod
    def _tokenize_gloss(gloss_seq: str) -> list:
        stem_replace = re.sub("[A-Z]?[a-z]+", "[STEM]", gloss_seq)
        words = stem_replace.split(" ")
        words = [re.sub("([.-]|~)", " \g<1> ", w) for w in words]
        return words

    def _tokenize_entry(self, entry: dict) -> dict:
        src_tokens = self._tokenize_src(entry["src"])
        src_tokens = self._augment_entry_tokens(src_tokens)
        gloss_tokens = self._tokenize_gloss(entry["gloss"])
        return {"src": src_tokens, "gloss": gloss_tokens}

    def _augment_entry_tokens(self, src_tokens: list) -> list:
        front_pad = ["<START>" for i in range(self.window_size)]
        back_pad = ["<END>" for i in range(self.window_size)]
        padded_seq = front_pad + src_tokens + back_pad

        windowed_seq = []
        for i in range(self.window_size, len(src_tokens) + self.window_size):
            windowed_seq.append(padded_seq[i-self.window_size:i+self.window_size+1])
        
        return windowed_seq

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

    def get_dev_covered_path(self):
        lang_folder = self.train_orig_path.parent
        return lang_folder / f"{self.language}-dev-track1-covered"

    def get_dev_uncovered_path(self):
        lang_folder = self.train_orig_path.parent
        return lang_folder / f"{self.language}-dev-track1-uncovered"
