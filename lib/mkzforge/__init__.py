import kizano
kizano.Config.setAppName('mkzforge')

from logging import Logger
def getLogger(n: str, ll: str = None, lf: str = 'standard') -> Logger: return kizano.getLogger(n, ll, lf)

import mkzforge.types as types
import mkzforge.const as const
import mkzforge.cli as cli
import mkzforge.genimg as genimg
import mkzforge.grive as grive
import mkzforge.metadata as metadata
import mkzforge.subtitles as subtitles
import mkzforge.i18n as i18n
import mkzforge.notify as notify
import mkzforge.videos as videos
import mkzforge.webserv as webserv

def main():
    '''
    Main entry point for this application.
    Let's you run commands for mkzforge.
    '''
    kizano.log.setLevel(99)
    config = kizano.getConfig()
    if 'google' in config and 'api_key' in config['google']:
        const.GOOGLE_API_KEY = config['google']['api_key']
    mkzforge = cli.Cli(config)
    mkzforge.execute()

__all__ = [
    'getLogger',
    'main',
    'types',
    'const',
    'cli',
    'genimg',
    'grive',
    'metadata',
    'subtitles',
    'i18n',
    'notify',
    'videos',
    'webserv',
]
