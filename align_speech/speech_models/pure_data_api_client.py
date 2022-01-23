import requests
import os
import io
from datetime import timedelta

from smart_open import open

import logging

logger = logging.getLogger(__name__)

class PureDataAPIClient:
    def __init__(self, config):
        self.config = config

    def predict(self, audio, name, label):

        logger.debug("Running pure data speech to text on: " + name)

        local_path = self.get_local_path(audio, name)

        response = requests.post(
            url=self.config["deploy"]["model"]["endpoint"],
            json={"path" : local_path})

        logger.debug(" result is: " + str(response.json()))

        # format the result to match google speech api
        result = PureDataResult(response.json(), self.config)

        return [result]

    def get_local_path(self, audio, name):
        gcs_path = os.path.join(self.config["deploy"]["model"]["storage_path"], name)

        with open(gcs_path, "wb") as gcs_file:
            with io.BytesIO() as temp_file:
                audio.export(temp_file, format="flac")
                gcs_file.write(temp_file.read())

        return gcs_path

class PureDataResult:
    def __init__(self, result, config):
        self.alternatives = [PureDataAlternative(result, config)]

class PureDataAlternative:
    def __init__(self, result, config):
        self.confidence = 1.0#result["score"]

        word_texts = result["label"].split()

        start_times = []
        end_times = []

        position = 0
        for word in word_texts:
            start_times.append(max(0.0, result["start_times"][position]+config["deploy"]["model"]["timestamp_offset"]))
            end_times.append(max(0.0, result["end_times"][position]+config["deploy"]["model"]["timestamp_offset"]))

            position += len(word) + 1

        self.words = [PureDataWord(word, start_time, end_time) for word, start_time, end_time in zip(word_texts, start_times, end_times)]

class PureDataWord:
    def __init__(self, word, start_time, end_time):
        self.word = word
        self.start_time = timedelta(seconds=start_time)
        self.end_time = timedelta(seconds=end_time)



