
import os
import io

import srt

from smart_open import open
from pydub import AudioSegment
from google.cloud import speech

from auto_align.util.config import setup_config
from auto_align.util.config import setup_logging

from datetime import timedelta

import logging

logger = logging.getLogger(__name__)

def fix(arguments):
    config = setup_config(arguments)
    setup_logging(config)

    data = load_data(config)

    utterances = extract_utterances(data, config)

    save_utterances(utterances, config)

def load_data(config):
    audio = AudioSegment.from_file(config["data"]["audio_path"], "flac")

    with open(config["data"]["srt_path"]) as srt_file:
        for subtitle in srt.parse(srt_file.read()):
            start = max(0, int(subtitle.start.total_seconds() * 1000) - config["data"]["padding_ms"])
            end   = min(int(subtitle.end.total_seconds() * 1000) + config["data"]["padding_ms"], len(audio))
            yield { "audio" : audio, "start" : start, "end" : end, "label" : subtitle.content }

def extract_utterances(data, config):
    model = GoogleSpeechAPIClient(config)

    for index, caption in enumerate(data):

        start = caption["start"]
        end = caption["end"]

        audio_segment = caption["audio"][start:end]

        name = str(index) + ".flac"

        result = model.predict(audio_segment, name)

        match = compare_captions(result, caption)

        logger.debug("Label is: " + str(caption["label"]))

        if match is None:
            continue

        logger.debug("Best match: " + str(match))

        yield match

class GoogleSpeechAPIClient:
    def __init__(self, config):
        self.config = config
        self.client = speech.SpeechClient()

    def predict(self, audio, name):

        logger.debug("Running google speech to text on: " + name)

        gcs_path = self.copy_to_gcs(audio, name)

        audio = speech.RecognitionAudio(uri=gcs_path)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=48000,
            enable_word_time_offsets=True,
            #enable_word_confidence=True,
            language_code=self.config["deploy"]["model"]["language"],
        )

        response = self.client.recognize(config=config, audio=audio)

        logger.debug(" result is: " + str(response.results))

        return response.results

    def copy_to_gcs(self, audio, name):
        gcs_path = os.path.join(self.config["deploy"]["model"]["google_cloud_storage_path"], name)

        with open(gcs_path, "wb") as gcs_file:
            with io.BytesIO() as temp_file:
                audio.export(temp_file, format="flac")
                gcs_file.write(temp_file.read())

        return gcs_path

def compare_captions(result, caption):
    if len(result) == 0:
        return None

    label_words = get_label_words(caption["label"])

    if len(label_words) == 0:
        return None

    start = caption["start"]

    best_match = None

    for alternative in result[0].alternatives:
        matched_start = False
        matched_end   = False

        # try to match the start
        for word in alternative.words:
            if word.word.lower() == label_words[0]:
                matched_start = True
                start_time = start + (word.start_time.total_seconds() * 1e3)
                break

        # try to match the end
        for word in reversed(alternative.words):
            if word.word.lower() == label_words[-1]:
                matched_end = True
                end_time = start + (word.end_time.total_seconds() * 1e3)
                break

        if matched_start and matched_end:
            old_confidence = 0 if best_match is None else best_match["confidence"]

            if alternative.confidence > old_confidence:
                best_match = {
                    "confidence" : alternative.confidence,
                    "start" : start_time,
                    "end" : end_time,
                    "label" : caption["label"]
                }

    return best_match

def get_label_words(label):
    words = label.split()

    return [word for word in words if not is_punctuation(word)]

def is_punctuation(word):
    return word == "." or word == ","

def save_utterances(utterances, config):
    subtitles = []
    for index, utterance in enumerate(utterances):
        start_time = timedelta(milliseconds=utterance["start"])
        end_time = timedelta(milliseconds=utterance["end"])
        subtitles.append(srt.Subtitle(index=index, start=start_time, end=end_time, content=utterance["label"]))

    with open(config["data"]["result_srt_path"], "w") as output_file:
        output_file.write(srt.compose(subtitles))



