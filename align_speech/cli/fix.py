from argparse import ArgumentParser

from align_speech.core.fix import fix

def main():

    parser = ArgumentParser(description="Fix the alignments using the start and end times from a speech model.")

    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    fix(arguments)

################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################


