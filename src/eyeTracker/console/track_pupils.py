# Builtin dependencies
import argparse
import logging

# Local dependencies
from eyeTracker.eyeTracker import eyeTracker


def _main(args):
    """
    Actual script (without command line parsing).

    It instances an eyeTracker object, and calls the startTrackingPupils method.

    Parameters
    ----------
    args: namespace object as returned by ArgumentParser.parse_args()
    """

    object = eyeTracker()
    object.startTracking(track_pupils=True, source=args['file'], show=args['show'], output=args['output'])

    return


def main():
    """Usage: eye-tracker-track-pupils FILE [-s] [-o] OUTPUT_FILE

    Arguments
    ----------
    "-f" : "--file"
        Video File
    "-s" : "--show"
        Show debug (default=True)
    "-o" : "--output"
        Output File (default=None)

    """

    # Args parser
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", help="Video File")
    argparser.add_argument("-s", "--show", help="Show debug", default=True)
    argparser.add_argument("-o", "--output", help="Output File", default=None)
    argparser.add_argument("-v", "--verbose", help="Increase logging output "
                            "(can be specified several times)", action="count", default=0)
    args = vars(argparser.parse_args())

    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    _V_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]
    loglevel = min(len(_V_LEVELS)-1, args['verbose'])
    logging.basicConfig(format=FORMAT, level = _V_LEVELS[loglevel])

    r = _main(args)

    logging.shutdown()

    return r

if __name__ == '__main__':
    exit(main())
