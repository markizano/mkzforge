from enum import StrEnum

class Devices(StrEnum):
    '''
    Poor man's enumeration() object for device types.
    '''
    CPU = 'cpu'
    GPU = 'cuda'
    CUDA = 'cuda'
    AUTO = 'auto'

class Action(StrEnum):
    NEW = 'new'
    BUILD = 'build'
    CUT_SILENCE = 'cut-silence'
    GENSUBS = 'gensubs'
    GENIMG = 'genimage'
    LOADMOD = 'load-module'
    METADATA = 'metadata'
    PUBLISH = 'publish'
    SERVE = 'serve'
    MP4TOMKV = 'mp4-to-mkv'
    NORMALIZE = 'normalize'

class WhisperTask(StrEnum):
    TRANSCRIBE = 'transcribe'
    TRANSLATE = 'translate'

