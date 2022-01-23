
from align_speech.util.get_label_words import get_label_words

import io
import os

from google.cloud import speech

from smart_open import open

import logging

logger = logging.getLogger(__name__)

class GoogleSpeechAPIClient:
    def __init__(self, config):
        self.config = config
        self.client = speech.SpeechClient()

    def predict(self, audio, name, label):

        logger.debug("Running google speech to text on: " + name)

        gcs_path = self.copy_to_gcs(audio, name)

        recognize_audio = speech.RecognitionAudio(uri=gcs_path)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=audio.frame_rate,
            enable_word_time_offsets=True,
            speech_contexts = [speech.SpeechContext(phrases=get_label_words(label))],
            #enable_word_confidence=True,
            language_code=self.config["deploy"]["model"]["language"],
        )

        response = self.client.recognize(config=config,
            audio=recognize_audio, timeout=15.0)

        #logger.debug(" result is: " + str(response.results))

        return response.results

    def copy_to_gcs(self, audio, name):
        gcs_path = os.path.join(self.config["deploy"]["model"]["google_cloud_storage_path"], name)

        with open(gcs_path, "wb") as gcs_file:
            with io.BytesIO() as temp_file:
                audio.export(temp_file, format="flac")
                gcs_file.write(temp_file.read())

        return gcs_path

