from transformers import DataProcessor, InputExample
import os
import csv
from tqdm import tqdm


class Sst2Processor(DataProcessor):

    def receiveMapTensor(self, map_tensor):
        return InputExample(
            map_tensor["idx"].numpy(),
            map_tensor["sentence"].numpy().decode("utf-8"),
            None,
            str(map_tensor["label"].numpy()),
        )

    def receiveTrEx(self, folder_data):
        return self.genEx(self._read_tsv(os.path.join(folder_data, "train.tsv")), "train")

    def receiveDevEx(self, folder_data):
        return self.genEx(self._read_tsv(os.path.join(folder_data, "dev.tsv")), "dev")

    def receiveTsEx(self, folder_data):
        return self.genTestEx(self._read_tsv(os.path.join(folder_data, "test.tsv")), "test")

    def receiveLbls(self):
        return ["0", "1"]
    @classmethod
    def receiveLblTxt(cls):
        return ["negative ,", "positive ,"]

    def genTestEx(self, lines, type_set):
        instances = []
        text_index = 1 if type_set == "test" else 0
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (type_set, i)
            text_a = line[0][2:]
            label = line[0][0]
            instances.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances

    def genEx(self, lines, type_set):
        instances = []
        text_index = 1 if type_set == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (type_set, i)
            text_a = line[text_index]
            label = None if type_set == "test" else line[1]
            instances.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances

class Sst5Processor(DataProcessor):
    def receiveMapTensor(self, map_tensor):
        return InputExample(
            map_tensor["idx"].numpy(),
            map_tensor["sentence"].numpy().decode("utf-8"),
            None,
            str(map_tensor["label"].numpy()),
        )

    def receiveTrEx(self, folder_data):
        return self.genEx(self._read_tsv(os.path.join(folder_data, "train.tsv")), "train")

    def receiveDevEx(self, folder_data):
        return self.genEx(self._read_tsv(os.path.join(folder_data, "dev.tsv")), "dev")

    def receiveTsEx(self, folder_data):
        return self.genEx(self._read_tsv(os.path.join(folder_data, "test.tsv")), "test")

    def receiveLbls(self):
        return ["0", "1", "2", "3", "4"]
    @classmethod
    def receiveLblTxt(cls):
        return ["disgusted ,", "dissatisfied ,", "indifferent ,", "acceptable ,", "satisfied ,"]

    def genEx(self, lines, type_set):
        instances = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (type_set, i)
            text_a = line[1]
            label = line[0]
            instances.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TrecProcessor(DataProcessor):

    def receiveMapTensor(self, map_tensor):
        return InputExample(
            map_tensor["idx"].numpy(),
            map_tensor["sentence"].numpy().decode("utf-8"),
            None,
            str(map_tensor["label"].numpy()),
        )

    def _read_csv(self, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=","))

    def receiveTrEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "train.csv")), "train")

    def receiveDevEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "test.csv")), "dev")

    def receiveTsEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "test.csv")), "test")

    def receiveLbls(self):
        return ["0","1", "2", "3", "4", "5"]
    @classmethod
    def receiveLblTxt(cls):
        return ["Description ,", "Entity ,", "Abbreviation ,", "Human ,", "Location ,", "Number ,"]

    def genEx(self, lines, type_set):
        instances = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (type_set, i)
            text_a = line[1]
            label = line[0]
            instances.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances


class AGNewsProcessor(DataProcessor):

    def receiveMapTensor(self, map_tensor):
        return InputExample(
            map_tensor["idx"].numpy(),
            map_tensor["sentence"].numpy().decode("utf-8"),
            None,
            str(map_tensor["label"].numpy()),
        )

    def _read_csv(self, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=","))

    def receiveTrEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "train.csv")), "train")

    def receiveDevEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "test.csv")), "dev")

    def receiveTsEx(self, folder_data):
        return self.genEx(self._read_csv(os.path.join(folder_data, "test.csv")), "test")

    def receiveLbls(self):
        return ["1", "2", "3", "4"]
    @classmethod
    def receiveLblTxt(cls):
        return ["World ,", "Sports ,", "Business ,", "science and technology ,"]

    def genEx(self, lines, type_set):
        instances = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (type_set, i)
            text_a = " ".join(line[1:])
            label = line[0]
            instances.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return instances


data_processors = {
    "sst-2": Sst2Processor,
    "sst-5": Sst5Processor,
    "trec": TrecProcessor,
    "agnews": AGNewsProcessor,
}


data_modes = {
    "sst-2": "classification",
    "sst-5": "classification",
    "trec": "classification",
    "agnews": "classification",
}

data_nums = {
    "sst-2": 2,
    "sst-5": 5,
    "trec": 6,
    "agnews": 4,
}