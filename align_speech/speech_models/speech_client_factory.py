
from align_speech.speech_models.google_speech_api_client import GoogleSpeechAPIClient
from align_speech.speech_models.pure_data_api_client import PureDataAPIClient

class SpeechClientFactory:
    def __init__(self, config):
        self.config = config

    def create(self):
        if self.config["deploy"]["model"]["type"] == "google":
            return GoogleSpeechAPIClient(self.config)
        elif self.config["deploy"]["model"]["type"] == "conformer":
            return PureDataAPIClient(self.config)

        assert False, "Not implemented"


