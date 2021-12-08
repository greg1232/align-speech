# Autoalign

Attempts to fix start and end timestamps in SRT files using the timestamps from
a speech to text model.  The current version uses google cloud speech.

# Installation:

1. Checkout the code.

```shell
  git clone git@github.com:greg1232/auto-align.git
```

2. Run it

```shell
  ./fix
```

# Paths

Change the input and output files in the [config file](auto_align/configs/peoples_speech_dev_set.yaml)

# Credentials

Setup credentials to use the google cloud speech API.  Put them in a file: ${HOME}/.aws/google-cloud-credentials.json


