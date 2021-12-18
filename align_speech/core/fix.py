
import srt

from pydub import AudioSegment

from align_speech.util.config import setup_config
from align_speech.util.config import setup_logging

from datetime import timedelta

from align_speech import align

import logging

logger = logging.getLogger(__name__)

def fix(arguments):
    config = setup_config(arguments)
    setup_logging(config)

    utterances = load_utterances(config)

    aligned_utterances = align(utterances, config)

    save_utterances(aligned_utterances, config)

def load_utterances(config):
    audio = AudioSegment.from_file(config["data"]["audio_path"], "flac")

    with open(config["data"]["srt_path"]) as srt_file:
        for subtitle in srt.parse(srt_file.read()):
            start = max(0, int(subtitle.start.total_seconds() * 1000) - config["data"]["padding_ms"])
            end   = min(int(subtitle.end.total_seconds() * 1000) + config["data"]["padding_ms"], len(audio))
            yield { "audio" : audio, "start" : start, "end" : end, "max_length" : len(audio), "label" : subtitle.content }

def save_utterances(utterances, config):
    subtitles = []
    for index, utterance in enumerate(utterances):
        start_time = timedelta(milliseconds=utterance["start"])
        end_time = timedelta(milliseconds=utterance["end"])
        subtitles.append(srt.Subtitle(index=index, start=start_time, end=end_time, content=utterance["label"]))

    with open(config["data"]["result_srt_path"], "w") as output_file:
        output_file.write(srt.compose(subtitles))



