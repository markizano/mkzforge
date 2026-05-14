'''
Takes the place of the "refresh" function. Normalizes the input video and add to the configuration.
This is a change from "refresh".
The command line operation is different from the in-app functions in that they discover the
environment, collect config and CLI args and pass that intelligently into the functions.
The functions themselves may perform config-changing operations or make calls to
`mkzforge build` internally to craft intermittent resources.
'''
import os
from mkzforge import getLogger, utils, videos, genimg, subtitles, metadata

log = getLogger(__name__)

def compressVideos(cfg: dict) -> int:
    '''
    Takes the inputs and runs the `mp4tomkv()` method with expected args.
    Minimize and compress videos via re-encoding them.
    '''
    log.info('Compressing resources to save on disk space...')
    mkzforge_cfg = utils.load()
    for resource in utils.getResources():
        if not resource.endswith('mp4'): continue
        if utils.hasInput(mkzforge_cfg, resource): continue
        log.info(f'Identified {resource} to be compressed...')
        compressed = videos.mp4tomkv(resource, **cfg)
        log.info(f'Compressed video {compressed}')
    log.info('Done compressing resources!')
    return 0

def cutSilence(cfg: dict) -> int:
    '''
    Find quiet moments in the video and cut/strip them out as configured.
    '''
    log.info('Cut-Silence from resources to remove dead moments...')
    mkzforge_cfg = utils.load()
    for resource in utils.getResources():
        videos.removeSilence(mkzforge_cfg, resource, **cfg)
    utils.save(mkzforge_cfg['videos'])
    log.info('Removed silence from videos!')
    return 0

def genSubs(cfg: dict) -> int:
    '''
    Entrypoint to generate subtitles from the videos in the `resources/` folder.
    '''
    log.info('Generating subtitles for resources...')
    mkzforge_cfg = utils.load()
    for resource in utils.getResources():
        idx = utils.getInputIndex(mkzforge_cfg['videos'], resource)
        if idx is None:
            log.info(f'Resource {resource} not found in mkzforge.yml config. Generating new config.')
            video_cfg = videos.newVideo(resource)
        else:
            log.info(f'Found {resource} in mkzforge.yml.')
            video_cfg = mkzforge_cfg['videos'][idx]
        subtitles.genSubtitles(video_cfg, resource, **cfg)
        mkzforge_cfg['videos'].append(video_cfg)
    utils.save(mkzforge_cfg['videos'])
    log.info('Done generating subtitles!')
    return 0

def genMetadata(cfg: dict) -> int:
    '''
    Produce a title and description based on content of the video.
    Requires/Assumes `genSubs()` has already been called.
    '''
    log.info('Generating title and description metadata.')
    mkzforge_cfg = utils.load()
    for video_cfg in mkzforge_cfg['videos']:
        if 'subs' not in video_cfg['attributes']: continue
        metadata.generateMetadata(video_cfg, 'title', **cfg)
        metadata.generateMetadata(video_cfg, 'description', **cfg)
    utils.save(mkzforge_cfg['videos'])
    log.info('Done generating title and description metadata.')
    return 0

def genImage(cfg: dict) -> int:
    '''
    Main entrypoint to generate an image.
    Creates a thumbnail from the video content used in the gen-image prompt.
    Requires/Assumes `genSubs()` and `genMetadata()` have already been called here.
    '''
    log.info('Generating video thumbnail...')
    mkzforge_cfg = utils.load()
    for video_cfg in mkzforge_cfg['videos']:
        if 'thumbnail' not in video_cfg['attributes']: continue
        if os.path.exists('thumbnail.png') and not cfg.get('overwrite', False):
            log.info('Not overwriting thumbnail.png. SKIP')
            continue
        log.info('Generating thumbnail.png...')
        content = open(f'build/{mkzforge_cfg["name"]}.txt')
        genimg.generate_thumbnail(video_cfg['metadata']['title'], content)
    log.info('Done generating thumbnail image(s)!')
    return 0

def normalize(cfg: dict) -> int:
    '''
    Main method entrypoint.

    Operations include:
        * mp4tomkv: Convert and compress video.
        * cut-silence: Strip the video of pauses in audio.
        * gensubs: Generate subtitles for video.
        * metadata: Generate metadata (title and description).
        * amend/configure: Add this video to the `mkzforge.yml` config for compilation.
    '''
    log.info('Normalizing videos to produce compressed subtitled videos with metadata attached.')
    # There's no video config to consider. So determine language as soon as possible.
    mkzforge_cfg = utils.load()
    cfg['name'] = os.path.basename(os.getcwd())
    video_cfg, resource = videos.detectState(mkzforge_cfg, **cfg)
    subtitles.genSubtitles(video_cfg, resource, **cfg)
    metadata.generateMetadata(video_cfg, 'description', **cfg)
    metadata.generateMetadata(video_cfg, 'title', **cfg)
    # Set filter_complex to None to get the default hardsub filter.
    videos.updateVideo(video_cfg, attributes=['thumbnail'], filter_complex=None)
    content = open(f'build/{utils.filename(resource)}.txt').read()
    if not os.path.exists('thumbnail.png') or ( os.path.exists('thumbnail.png') and cfg.get('overwrite', False) ):
        genimg.generate_thumbnail(video_cfg['metadata']['title'], content)

    if not utils.hasInput(mkzforge_cfg['videos'], resource):
        mkzforge_cfg['videos'].append(video_cfg)
    log.info('Video(s) normalized and added to `mkzforge.yml` config.')
    utils.save(mkzforge_cfg['videos'])
    log.info('Done normalizing videos!')
    return 0
