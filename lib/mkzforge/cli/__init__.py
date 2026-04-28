'''
Command line interface entrypoint for `mkzforge` command.

This module will be responsible for parsing command line arguments and
calling the appropriate sub-modules to perform the actions requested.
'''

import os, sys
from signal import signal, SIGINT
from multiprocessing import cpu_count
from argparse import ArgumentParser, RawTextHelpFormatter
from kizano import getLogger
log = getLogger(__name__)

from mkzforge import types, const
import mkzforge.cli.new as new
import mkzforge.cli.normalize as normalize
import mkzforge.cli.build as build
import mkzforge.cli.publish as publish
import mkzforge.cli.web as web

def interrupt(signal, frame):
    log.error('Caught ^C interrupt, exiting...')
    sys.exit(signal)

class Cli(object):
    '''
    Usage: %(prog)s [options] [command]
    '''

    ACTIONS = {
        types.Action.NEW: new.gennew,
        types.Action.CUT_SILENCE: normalize.cutSilence,
        types.Action.MP4TOMKV: normalize.compressVideos,
        types.Action.GENSUBS: normalize.genSubs,
        types.Action.GENIMG: normalize.genImage,
        types.Action.METADATA: normalize.genMetadata,
        types.Action.NORMALIZE: normalize.normalize,
        types.Action.BUILD: build.buildVideo,
        types.Action.PUBLISH: publish.publishBuilds,
        types.Action.SERVE: web.serveTheWeb,
    }

    def __init__(self, config: dict):
        self.config = config
        self.getOptions()
        log.debug(f'Final config: {self.config}')

    def getOptions(self) -> None:
        '''
        Gets command line arguments.
        '''
        options = ArgumentParser(
            usage=self.__doc__,
            formatter_class=RawTextHelpFormatter
        )

        options.add_argument(
            '--log-level', '-l',
            action='store',
            dest='log_level',
            help='How verbose should this application be?',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            type=str.upper
        )

        options.add_argument(
            '--version',
            action='store_true',
            dest='print_version',
            help='Print version and exit.',
        )

        subparsers = options.add_subparsers(dest='action', metavar='command')

        # new
        new_parser = subparsers.add_parser(
            str(types.Action.NEW),
            help='Create new project structure.',
            formatter_class=RawTextHelpFormatter
        )
        new_parser.add_argument(
            'resource',
            nargs='?',
            help='Optional directory path for the new project.',
            default=None
        )

        # mp4-to-mkv
        mp4tomkv_parser = subparsers.add_parser(
            str(types.Action.MP4TOMKV),
            help='Convert and compress MP4 files to MKV format.',
            formatter_class=RawTextHelpFormatter
        )
        mp4tomkv_parser.add_argument(
            '--force', '-f',
            action='store_true',
            dest='overwrite',
            help='Runs a cleanup of files before writing new files.',
            default=None
        )
        mp4tomkv_parser.add_argument(
            '--no-subtitles',
            action='store_false',
            dest='subtitles',
            help='Do not manage subtitle streams.',
            default=None
        )
        mp4tomkv_parser.add_argument(
            '--silence-detect',
            action='store',
            dest='cut_silence',
            type=bool,
            help='Enables silence detection and removal (Default: True).',
            default=True,
        )
        mp4tomkv_parser.add_argument(
            '--silence-threshold',
            action='store',
            dest='silence_threshold',
            help='Silence threshold in decibels (default: 30).',
            type=int,
        )
        mp4tomkv_parser.add_argument(
            '--silence-duration',
            action='store',
            dest='silence_duration',
            help='Minimum silence duration in seconds (default: 1.2).',
            type=float,
        )
        mp4tomkv_parser.add_argument(
            '--silence-pad',
            action='store',
            dest='silence_pad',
            help='Padding in milliseconds before/after silence removal (default: 350).',
            type=int,
            default=350
        )

        # gensubs
        gensubs_parser = subparsers.add_parser(
            str(types.Action.GENSUBS),
            help='Generate subtitles for video files.',
            formatter_class=RawTextHelpFormatter
        )
        gensubs_parser.add_argument(
            'resource',
            nargs='?',
            help='Specific video file to generate subtitles for.',
            default=None
        )
        gensubs_parser.add_argument(
            '--device', '-d',
            action='store',
            dest='device',
            help='Specify the device to use for processing.',
            choices=[types.Devices.CPU, types.Devices.CUDA, types.Devices.AUTO],
            default=types.Devices.GPU
        )
        gensubs_parser.add_argument(
            '--language',
            action='store',
            dest='language',
            help='Which language to use when generating subtitles?',
            choices=const.LANGS,
            default=None
        )
        gensubs_parser.add_argument(
            '--no-subtitles',
            action='store_false',
            dest='subtitles',
            help='Do not manage subtitle streams.',
            default=None
        )

        # genimage
        genimage_parser = subparsers.add_parser(
            str(types.Action.GENIMG),
            help='Generate thumbnail images from video content.',
            formatter_class=RawTextHelpFormatter
        )
        genimage_parser.add_argument(
            '--force', '-f',
            action='store_true',
            dest='overwrite',
            help='Overwrite existing thumbnail.',
            default=None
        )

        # metadata
        metadata_parser = subparsers.add_parser(
            str(types.Action.METADATA),
            help='Generate video metadata (title and description) from subtitles.',
            formatter_class=RawTextHelpFormatter
        )
        metadata_parser.add_argument(
            '--no-title',
            action='store_false',
            dest='title',
            help="Don't generate a title.",
            default=None
        )
        metadata_parser.add_argument(
            '--no-description',
            action='store_false',
            dest='description',
            help="Don't generate a description.",
            default=None
        )

        # normalize (replaces refresh: mp4tomkv + silence removal + gensubs + metadata + config update)
        normalize_parser = subparsers.add_parser(
            str(types.Action.NORMALIZE),
            help='Normalize video: convert, remove silence, generate subtitles and metadata.',
            formatter_class=RawTextHelpFormatter
        )
        normalize_parser.add_argument(
            'resource',
            nargs='?',
            help='Specific video file to normalize.',
            default=None
        )
        normalize_parser.add_argument(
            '--force', '-f',
            action='store_true',
            dest='overwrite',
            help='Overwrite existing output files.',
            default=None
        )
        normalize_parser.add_argument(
            '--device', '-d',
            action='store',
            dest='device',
            help='Specify the device to use for processing.',
            choices=[types.Devices.CPU, types.Devices.CUDA, types.Devices.AUTO],
            default=types.Devices.GPU
        )
        normalize_parser.add_argument(
            '--language',
            action='store',
            dest='language',
            help='Which language to use when generating subtitles?',
            choices=const.LANGS,
            default=None
        )
        normalize_parser.add_argument(
            '--no-subtitles',
            action='store_false',
            dest='subtitles',
            help='Do not manage subtitle streams.',
            default=None
        )
        normalize_parser.add_argument(
            '--silence-detect',
            action='store',
            dest='cut_silence',
            type=bool,
            help='Enables silence detection and removal (Default: True).',
            default=True,
        )
        normalize_parser.add_argument(
            '--silence-threshold',
            action='store',
            dest='silence_threshold',
            help='Silence threshold in decibels (default: 30).',
            type=int,
        )
        normalize_parser.add_argument(
            '--silence-duration',
            action='store',
            dest='silence_duration',
            help='Minimum silence duration in seconds (default: 1.2).',
            type=float,
        )
        normalize_parser.add_argument(
            '--silence-pad',
            action='store',
            dest='silence_pad',
            help='Padding in milliseconds before/after silence removal (default: 350).',
            type=int,
            default=350
        )
        normalize_parser.add_argument(
            '--no-title',
            action='store_false',
            dest='title',
            help="Don't generate a title.",
            default=None
        )
        normalize_parser.add_argument(
            '--no-description',
            action='store_false',
            dest='description',
            help="Don't generate a description.",
            default=None
        )

        # build
        build_parser = subparsers.add_parser(
            str(types.Action.BUILD),
            help='Build final video from configuration.',
            formatter_class=RawTextHelpFormatter
        )
        build_parser.add_argument(
            'resource',
            nargs='?',
            help='Output file path.',
            default=None
        )
        build_parser.add_argument(
            '--no-autoplay',
            action='store_false',
            dest='autoplay',
            help="Don't attempt to automatically play the video after building.",
            default=None
        )

        # publish
        publish_parser = subparsers.add_parser(
            str(types.Action.PUBLISH),
            help='Publish compiled video to configured endpoints (SFTP, YouTube, TikTok).',
            formatter_class=RawTextHelpFormatter
        )
        publish_parser.add_argument(
            'resource',
            nargs='?',
            help='Specific output file to publish (publishes all if omitted).',
            default=None
        )

        # serve
        serve_parser = subparsers.add_parser(
            str(types.Action.SERVE),
            help='Start the web interface.',
            formatter_class=RawTextHelpFormatter
        )
        serve_parser.add_argument(
            '--workspace',
            action='store',
            dest='workspace',
            help='Workspace directory for video projects.',
            default=None
        )
        serve_parser.add_argument(
            '--http-port',
            action='store',
            dest='http_port',
            help='HTTP port for web server (default: 9091).',
            type=int,
            default=None
        )
        serve_parser.add_argument(
            '--webroot',
            action='store',
            dest='webroot',
            help='Directory containing web assets.',
            default=None
        )

        opts = options.parse_args()

        if opts.print_version:
            import mkzforge._version
            print(f'mkzforge: {mkzforge._version.__version__}')
            sys.exit(0)

        if not 'LOG_LEVEL' in os.environ:
            os.environ['LOG_LEVEL'] = opts.log_level
            log.setLevel(opts.log_level)

        if not opts.action:
            log.error('No action specified!')
            options.print_help()
            return

        # Filter out None values so config file defaults are not overridden.
        opts_dict = {k: v for k, v in vars(opts).items() if v is not None}
        self.config.update(opts_dict)
        # The following "defaults" are set **after** everything above because if a default is
        # defined in the ArgumentParser(), it does not allow ~/.config/mkzforge/config.yml
        # to define the default behaviour/config. Setting it here ensures that is possible
        # and if still not defined (e.g. config absent), we can define in-code.
        self.config['silence_duration'] = self.config.get('silence_duration', 1.2)
        self.config['silence_threshold'] = self.config.get('silence_threshold', 30)

    def execute(self):
        '''
        Interprets command line options and calls the subsequent actions to take.
        These will be built out as sub-modules to this module.
        '''
        log.info('Welcome to mkzforge!')
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(cpu_count())
        action = Cli.ACTIONS.get(self.config['action'])
        if not action:
            log.error('Invalid action: %s', self.config['action'])
            return 1
        log.info(f'Executing action: {action} with config: {self.config}')
        signal(SIGINT, interrupt)
        return action(self.config)

__all__ = [
    'base',
    'new',
    'compress',
    'genimage',
    'metadata',
    'normalize',
    'build',
    'publish',
    'web',
    'Cli',
]
