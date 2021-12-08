
import config
import os
import logging

logger = logging.getLogger(__name__)

def setup_config(dictionary = {}):
    return config.ConfigurationSet(
        config.config_from_env(prefix="PEOPLES_SPEECH", separator="_", lowercase_keys=True),
        config.config_from_yaml(config_path(), read_from_file=True),
        config.config_from_dict(dictionary),
    )

def config_path():
    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".peoples_speech_config.yaml")
    if os.path.exists(home_config_path):
        return home_config_path

    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "peoples_speech_dev_set.yaml")

def setup_logging(arguments):

    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("smart_open").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)


