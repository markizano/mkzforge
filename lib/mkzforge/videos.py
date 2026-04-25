'''
Video processing functions.
Input a path to a video, perform some operation on it, output the updated/transmuted video.
Some functions take a video path to inspect and output a filter complex to execute against it.
'''
import os
import sys
import re
import time
import ffmpeg
from ffmpeg.errors import FFmpegError
from copy import deepcopy as copy
from mkzforge import getLogger, utils
log = getLogger(__name__)

def newProject(resource: str, **kwargs) -> int:
    if not os.path.exists(resource):
        log.info(f"Creating new resource directory in {resource}.")
        os.makedirs(resource, exist_ok=True)
    else:
        if not os.path.isdir(resource):
            log.error(f"{resource} is not a directory!")
            return 1
    os.chdir(resource)
    log.info(f'Creating new project directory in {os.getcwd()}.')
    if not os.path.exists('build'):     os.mkdir('build')
    if not os.path.exists('resources'): os.mkdir('resources')
    if not os.path.exists('readme.md'): open('readme.md', 'w').write('')
    if not os.path.exists('mkzforge.yml'):
        utils.save([])
    log.info('Created new project directory in current working directory.')

def mp4tomkv(resource: str) -> str:
    '''
    Convert an MP4 file to MKV.
    Strip metadata and video sidedata.
    Compress using crf=28.
    '''
    log.info(f'Converting {resource} to mkv.')
    if resource.endswith('.mkv'):
        log.info('Video already MKV, not converting!')
        return resource
    mkvfile = resource.replace('.mp4', '.mkv')
    out_opts = {
        'vf': 'sidedata=mode=delete',
        'f': 'matroska',
        'c:v': 'libx265',
        'c:a': 'ac3',
        'crf': 28,
        'pix_fmt': 'yuv420p',
        'map_metadata': '-1',
        'metadata:s:a:0': 'language=eng',
    }
    stream = ffmpeg.FFmpeg(executable=os.getenv('FFMPEG_BIN', 'ffmpeg'))
    stream.option('hide_banner')
    stream.option('y')
    stream.input(resource)
    stream.output(mkvfile, **out_opts)
    stream.on('stderr')(log.debug)
    stream.execute()
    log.debug(f'Deleting {resource} to save on disk space.')
    os.unlink(resource)
    return mkvfile

def getVideoDuration(resource: str) -> float:
    '''
    Get the duration of a video file in seconds.
    '''
    # This ffprobe will output the number of seconds in the video.
    probe_stream = (
        ffmpeg.FFmpeg(executable=os.getenv('FFPROBE_BIN', "ffprobe"))
        .option('v', 'quiet')
        .option('show_entries', 'format=duration')
        .option('of', 'csv=p=0')
        .input(resource)
    )

    return float(probe_stream.execute().decode('utf-8').strip())

def getVideoRotation(resource: str) -> int:
    '''
    Detect video rotation from display matrix metadata using ffprobe.
    Returns the rotation angle in degrees (0, 90, 180, 270).
    '''
    log.info(f'Detecting rotation for {resource}...')
    try:
        rotation_output = []

        # Use ffprobe to get rotation metadata
        probe_stream = (
            ffmpeg.FFmpeg(executable=os.getenv('FFPROBE_BIN', "ffprobe"))
            .option('v', 'quiet')
            .option('show_entries', 'stream_tags=rotate:stream_side_data=rotation')
            .option('of', 'csv=p=0')
            .input(resource)
        )

        rotation_output = probe_stream.execute().decode('utf-8')

        # Parse rotation value
        if rotation_output:
            rotation = int(rotation_output.strip().strip(','))
            log.info(f'Detected rotation: {rotation} degrees for {resource}')
            return rotation
        log.debug(f'No rotation metadata found for {resource}')
        return 0

    except Exception as e:
        log.warning(f'Could not detect rotation for {resource}: {e}')
        return 0

def detectSilence(resource: str, **kwargs) -> list[str]:
    '''
    Detect silence in a video file and return filter_complex strings to remove silent segments.
    Params:

        resource: The video file in question to detect silence. Usually out of the `resources/`
            folder.
        silence_threshold: What sound level in dB to count as "silence"
        silence_duration: How long in seconds should we count said silence to be clipped.
        silence_pad: Silence stripping can be agressive. Pad/restore the before & after silence by
            this many miliseconds of video to make the clips seem more natural/clean.

    Returns: List of filter complex strings representing silence removal
    '''
    log.info(f'Detecting silence in \x1b[1m{resource}\x1b[0m...')
    silence_threshold: int = kwargs.get('silence_threshold', 30)
    silence_duration: float = kwargs.get('silence_duration', 1.2)
    silence_pad_ms: int = kwargs.get('silence_pad_ms', 350)

    # Use FFmpeg silencedetect filter to find silent segments
    silence_output = []
    log.info(f'Using silence threshold: -{silence_threshold}dB, duration: {silence_duration}s')

    try:
        out_args = {
            'vn': None,
            'f': 'null',
            'af': f'silencedetect=noise=-{silence_threshold}dB:d={silence_duration}',
        }
        stream = (
            ffmpeg.FFmpeg()
            .option('hide_banner')
            .input(resource)
            .output('-', **out_args)
        )

        @stream.on('stderr')
        def on_stderr(line):
            silence_output.append(line)

        stream.execute()
        output = '\n'.join(silence_output)
    except Exception as e:
        log.error(f'Error running silence detection: {e}')
        return []

    # Parse silence detection output
    silence_start_pattern = r'silence_start: ([\d.]+)'
    silence_end_pattern = r'silence_end: ([\d.]+)'

    silence_starts = [round(float(match), 3) for match in re.findall(silence_start_pattern, output)]
    silence_ends = [round(float(match), 3) for match in re.findall(silence_end_pattern, output)]

    if not silence_starts or not silence_ends:
        log.info('No significant silence detected.')
        return []

    # Get padding value in seconds (from milliseconds)
    silence_pad = silence_pad_ms / 1000.0  # Convert ms to seconds
    log.info(f'Using silence padding: {silence_pad}s ({silence_pad_ms}ms)')

    # Create segments to keep (non-silent parts)
    segments = []
    current_time = 0.0
    total_duration = getVideoDuration(resource)
    log.info(f'Total video duration: {total_duration}')

    for i, start in enumerate(silence_starts):
        # Add segment before silence, extending into the silence by the padding amount
        if start > current_time:
            segment_start = current_time
            segment_end = round(min(start + silence_pad, total_duration), 3)
            segments.append((segment_start, segment_end))

        # Update current time to end of silence (if available), backing up by padding amount
        if i < len(silence_ends):
            current_time = round(max(0.0, silence_ends[i] - silence_pad), 3)

    # Add final segment if there's content after the last silence
    if current_time < total_duration:
        segments.append((current_time, total_duration))

    if not segments:
        log.info('No segments to keep after silence removal.')
        return []

    log.info(f'Found {len(segments)} segments to keep after removing silence.')
    log.debug(f'Segments: {segments}')

    # Generate filter_complex strings for trimming and concatenating segments
    trim_filters = []
    for i, (start, end) in enumerate(segments):
        trim_filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,setsar=1:1[v{i}]")
        trim_filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")

    # Create concat filter for video and audio
    trim_filters.extend(utils.mergefilters(segments))
    log.info(f'Done detecting silence for {resource}!')
    return trim_filters

def removeSilence(mkzforge_cfg: dict, resource: str, **kwargs) -> str:
    '''
    Process a video to remove silent segments and output to build/ directory.
    If the input is MP4, also handles conversion to MKV with CRF 28.
    Returns the path to the trimmed video.
    '''
    # Get silence filters
    silence_filters = detectSilence(resource, **kwargs)
    if not silence_filters:
        log.info('No silence detected, using original video.')
        return resource

    video_cfg = newVideo(resource)
    # Avoid clashing names with the target MP$4 video for the final build.
    # Also, this is still an "internal" resource, just in the build directory.
    output = f'build/{utils.filename(resource)}.mkv'
    updateVideo(video_cfg, output=output, filter_complex=silence_filters)
    mkzforge_cfg['videos'].append(video_cfg)
    log.info(f'Processing silence removal for \x1b[1m{resource}\x1b[0m...')
    cv = compileVideo(video_cfg, **kwargs)
    if cv != 0:
        log.warning('compileVideo() at this step did not succeed. This may cause downstream effects from here...')
    else:
        log.info(f'Successfully created silence-trimmed video at \x1b[1m{output}\x1b[0m!')
    return output

def newVideo(resource: str) -> dict:
    '''
    Generate a video config to append to the mkzforge.yml configuration.
    If title or description are empty in the config defaults, they will be auto-generated from
    subtitles.

    Returns: The video configuration you can add to `mkzforge.yml`.
    '''
    log.info(f'Generating \x1b[1m{resource}\x1b[0m to add to mkzforge.yml configuration.')
    return {
        'input': [
            { 'i': resource },
        ],
        'output': f'build/{utils.filename(resource)}.mp4',
        'metadata': {},
        'attributes': [],
        'movflags': ['+faststart'],
    }

def updateVideo(video_cfg: dict, **kwargs) -> dict:
    '''
    Once we have more details about a video, update the data structure accordingly.

    title: str -- Assign the metadata title.
    description: str -- Assign the metadata description.
    subs: dict[str, str] -- key-value mapping of language:filename for subtitles.
    filter_complex: list[str] -- List of strings to use for the filter complex. By default
        include the standard hard-sub filter complex.
    '''
    log.info(f'Updating video attributes and metadata: {video_cfg}')

    # If metadata is defined, then assign it where it belongs in the data structure.
    for metadata in ['title', 'description']:
        if metadata in kwargs and kwargs[metadata]:
            video_cfg['metadata'][metadata] = kwargs[metadata]

    if 'attributes' in kwargs and kwargs['attributes']:
        for attribute in kwargs['attributes']:
            if attribute not in video_cfg['attributes']:
                video_cfg['attributes'].append(attribute)

    if 'output' in kwargs and kwargs['output']:
        video_cfg['output'] = kwargs['output']

    if 'subs' in kwargs and kwargs['subs']:
        if 'subs' not in video_cfg['attributes']:
            video_cfg['attributes'].append('subs')
        if 'map' not in video_cfg:
            video_cfg['map'] = {}
        if 'languages' not in video_cfg:
            video_cfg['languages'] = []
        for idx, sub in enumerate(kwargs['subs']):
            lang, srtfile = sub
            langcode = f'{lang}:{idx}'
            if langcode not in video_cfg['languages']:
                video_cfg['languages'].append(langcode)
            i = len(video_cfg['input'])
            video_cfg['map'][lang] = f'{i}:s'
            if not utils.hasInput([video_cfg], srtfile):
                video_cfg['input'].append({'i': srtfile})

    if 'filter_complex' in kwargs:
        if kwargs['filter_complex']:
            video_cfg['filter_complex'] = kwargs['filter_complex']
        else:
            # Build video filter chain with rotation handling
            video_filters = []

            # Remember: https://www.abyssale.com/blog/how-to-change-the-appearances-of-subtitles-with-ffmpeg
            # @TODO: Detect when split screen OBS view and change font settings.
            font_style = ','.join([
                'Alignment=0',
                'PrimaryColour=&H00FFFFFF',
                'FontName=Impact',
                'OutlineColour=&H40000000',
                'BorderStyle=3',
                'Fontsize=14',
                'MarginV=60',
                'MarginL=30'
            ])
            # Get the first SRT file as input.
            srtfile = None
            for inputVid in video_cfg['input']:
                if inputVid['i'].endswith('.srt'):
                    srtfile = inputVid['i']
                    break
            if srtfile == None:
                raise ValueError(f'Input video has no SRT file: {video_cfg}')
            video_filters.append(f"subtitles={srtfile}:force_style='{font_style}'")

            # Build filter_complex with standard processing
            video_filter_str = ','.join(video_filters)
            video_cfg['filter_complex'] = [
                f"[0:v]{video_filter_str}[video]",
                # f"[0:a]volume=1.5,afftdn=nr=10:nf=-20:tn=1,equalizer=f=623:w=3.5:t=h:g=-15:n=1,asetpts=NB_CONSUMED_SAMPLES/SR/TB[audio]"
                f"[0:a]volume=1.5,asetpts=PTS-STARTPTS[audio]"
            ]

    log.info(f'New Video config: {video_cfg}')
    return video_cfg

def preProcessResources(mkzforge_cfg: dict, **kwargs) -> str:
    '''
    Handle the case of >1 resource.
    When we have >1 resource, need to:
    * mp4-to-mkv
    * cut-silence
    * concat the results.

    Return: Path to post-processed video path.
    '''
    to_concat: list[dict] = []
    concat_filter_complex = []
    for i, resource in enumerate(utils.getResources()):
        if utils.hasInput(mkzforge_cfg['videos'], resource): continue
        mp4tomkv(resource)
        trimmed_video = removeSilence(mkzforge_cfg, resource, **kwargs)
        to_concat.append({'i': trimmed_video})
        # Workaround to the merge filters util below.
        concat_filter_complex.append(f'[{i}:v]null[v{i}]')
        concat_filter_complex.append(f'[{i}:a]anull[a{i}]')
    # Once we are done iterating videos that need compress & cut, combine them.
    concat_filter_complex.extend(utils.mergefilters(to_concat))
    name = kwargs.get('name', os.path.basename(os.getcwd()))
    outfile = f'resources/{name}.mkv'
    if os.path.exists(outfile):
        log.warning(f'The {outfile} exists, writing to `resources/_{name}.mkv`')
        outfile = f'resources/_{name}.mkv'
    video_cfg = {
        'input': to_concat,
        'output': outfile,
        'filter_complex': concat_filter_complex,
        'metadata': {},
        'attributes': [],
        'movflags': ['+faststart'],
    }
    mkzforge_cfg['videos'].append(video_cfg)
    utils.save(mkzforge_cfg['videos'])
    cv = compileVideo(video_cfg, **kwargs)
    if cv != 0:
        log.warning('compileVideo() at this step did not succeed. This may cause downstream effects from here...')
    return outfile

def detectState(mkzforge_cfg: dict, **kwargs) -> tuple[dict, str]:
    '''
    Used in the `mkzforge normalize` function.
    Detect the current state of the environment based on the `mkzforge.yml` config and which
    resources are present on disk.
    Gets the active video configuration and the active resource we should be processing.
    '''
    mp4s = utils.getMP4s()
    if len(mp4s) >1:
        resource = preProcessResources(mkzforge_cfg, **kwargs)
    elif len(mp4s) == 1:
        mkv = mp4tomkv(mp4s[0])
        # cut-silence from video.
        resource = removeSilence(mkzforge_cfg, mkv, **kwargs)
        if utils.hasInput(mkzforge_cfg['videos'], resource):
            log.info(f'Resource {resource} found in config. Loading from this...')
            idx = utils.getInputIndex(mkzforge_cfg['videos'], resource)
            video_cfg = mkzforge_cfg['videos'][idx]
        else:
            log.info(f'Resource {resource} not found. Generating new config...')
            video_cfg = newVideo(resource)
    else:
        # It's safe to assume these will all be MKV files.
        log.info('No more MP4 to convert found. Checking MKV resources...')
        resources = utils.getResources()
        if len(resources) > 1:
            log.info('Multiple MKV files found.')
            if len(mkzforge_cfg['videos']) > 0:
                log.info('More than one MKV resource in config, checking for unseen MKV files.')
                unseen = False
                for res in resources:
                    if utils.hasInput(mkzforge_cfg['videos'], res): continue
                    log.info(f'Found {res} we have not yet processed. Working on this now.')
                    resource = res
                    video_cfg = newVideo(res)
                    unseen = True
                    break
                if not unseen:
                    log.info('We have at least processed all MKV files once. Checking to see if they have been concatenated...')
                    if utils.hasConcat(mkzforge_cfg['videos']):
                        video_cfg = mkzforge_cfg['videos'][-1]
                        resource = video_cfg['input'][0]['i']
                        log.info(f'Videos have been concat. Working with \x1b[1m{resource}\x1b[0m')
                    else:
                        log.info(f'Videos have NOT been concat. Combining to produce the artifact.')
                        resource = preProcessResources(mkzforge_cfg, **kwargs)
                        video_cfg = newVideo(resource)
            else:
                log.info('Combining MKV files since we have not added them to config yet.')
                resource = preProcessResources(mkzforge_cfg, **kwargs)
                video_cfg = newVideo(resource)
        elif len(resources) == 1:
            resource = resources[0]
            log.info(f'Found exactly one resource. Working with \x1b[1m{resource}\x1b[0m')
            idx = utils.getInputIndex(mkzforge_cfg['videos'], resource)
            if idx is None:
                log.info(f'Resource {resource} not found in mkzforge.yml config. Generating new config.')
                video_cfg = newVideo(resource)
            else:
                log.info(f'Found {resource} in mkzforge.yml.')
                video_cfg = mkzforge_cfg['videos'][idx]
    return video_cfg, resource

def _inputToFluidArgs(raw_input_video: str|dict) -> tuple[str, dict]:
    '''
    OLD function:
        Take the data structure that represents an input and convert into the argument list ffmpeg
        expects.
        Lead with -f for the format.
        Tail with -i for the input file.
        In this way, all options for that video are processed on the command line in the correct order.

    NEW function:
        Same option ordering as inputToArgList (dict branch), for ffmpeg.FFmpeg().input(..., **opts).

    Used to generate an array to pass to `subprocess` to run ffmpeg. Now we use the fluid interface.
    '''
    if isinstance(raw_input_video, str):
        log.debug(f'INPUT stream: \x1b[1m{raw_input_video}\x1b[0m')
        return raw_input_video, {}
    assert 'i' in raw_input_video, 'No input specified for video!'
    input_video = copy(raw_input_video)
    i = input_video.pop('i')
    log.debug(f'INPUT stream: \x1b[1m{i}\x1b[0m')
    opts: dict = {}
    while len(input_video):
        # Make sure the format is the first argument to be added (mirrors inputToArgList).
        if 'f' in input_video:
            opts['f'] = input_video.pop('f')
        # Add other arguments in the middle; popitem order matches the legacy argv builder.
        if len(input_video) >= 1:
            name, value = input_video.popitem()
            opts[name] = value
    return i, opts

def compileVideo(video_opts: dict, **kwargs) -> int:
    '''
    Craft the FFMpeg command and execute on the video conversion.
    '''
    with utils.video_processing_lock('build'):
        now = time.time()
        assert 'input' in video_opts, 'No input specified for video!'
        assert 'output' in video_opts, 'No output specified for video!'
        assert isinstance(video_opts['input'], list), 'Input must be an array!'
        vidnow = time.time()
        output = video_opts["output"]
        if output is None:
            log.warning(f'Skipping vid because output is set to None.')
            return 1
        attributes = video_opts.get('attributes', [])
        log.info(f'Processing video: \x1b[1m{output}\x1b[0m')
        language = video_opts.get('language', 'en')

        # Build the ffmpeg command with the fluid API (global opts, then each input in order).
        ffbin = os.getenv('FFMPEG_BIN', 'ffmpeg')
        final_cmd = ffmpeg.FFmpeg(executable=ffbin)
        final_cmd.option('hide_banner')
        final_cmd.option('y')
        output_opts: dict = {}
        # Map filter outputs or stream selectors onto the output file (video, audio, subs).
        map_streams: list[str] = []

        # Append each `-i` (with any per-input flags) in config order.
        for input_video in video_opts['input']:
            url, in_opts = _inputToFluidArgs(input_video)
            final_cmd.input(url, **in_opts)

        # Optional project thumbnail: extra input + disposition on the matching video stream index.
        if os.path.exists('thumbnail.png'):
            final_cmd.input('thumbnail.png')

        # filter_complex from script file; `-/filter_complex` matches Option key `/filter_complex`.
        if 'filter_complex' in video_opts:
            filter_complex = f'build/{utils.filename(output)}.filter_complex'
            filter_complex_script = ';\n'.join(fc_line for fc_line in video_opts['filter_complex'] if not fc_line.startswith('#')) + '\n'
            open(filter_complex, 'w', encoding='utf-8').write(filter_complex_script)
            final_cmd.option('/filter_complex', filter_complex)

        # Strip container metadata from inputs before we attach our own.
        output_opts['map_metadata'] = '-1'

        # Attach current required metadata (title, description, etc.).
        if 'metadata' in video_opts:
            output_opts['metadata'] = [
                f'{name}={value}' for name, value in video_opts['metadata'].items()
            ]

        if 'vsync' in attributes:
            output_opts['fps_mode'] = 'vfs'

        if 'no-video' in attributes:
            output_opts['vn'] = None
        else:
            map_streams.append(video_opts.get('map', {}).get('video', '[video]'))
            output_opts['c:v'] = video_opts.get('codecs', {}).get('video', 'h264')
            output_opts['pix_fmt'] = 'yuv420p'
            output_opts['crf'] = 28
            output_opts['metadata:s:v'] = f'language={language}'

        if 'no-audio' in attributes:
            output_opts['an'] = None
        else:
            map_streams.append(video_opts.get('map', {}).get('audio', '[audio]'))
            output_opts['c:a'] = video_opts.get('codecs', {}).get('audio', 'aac')
            # First output audio stream; matches behaviour of mp4tomkv metadata tagging.
            output_opts['metadata:s:a:0'] = f'language={language}'

        if 'subs' in attributes:
            output_opts['c:s'] = 'mov_text'
            if 'languages' in video_opts:
                # If you set languages and subs, map each language from `map.<lang>` (e.g. 1:s).
                for ilang in video_opts['languages']:
                    lang, i = ilang.split(':')
                    map_streams.append(video_opts['map'][lang])
                    output_opts[f'metadata:s:s:{i}'] = f'language={lang}'
            else:
                map_streams.append(video_opts.get('map', {}).get('subs', '[subs]'))
                output_opts['metadata:s:s'] = 'language=eng'

        if 'movflags' in video_opts:
            output_opts['movflags'] = video_opts['movflags']

        if map_streams:
            output_opts['map'] = map_streams

        if os.path.exists('thumbnail.png'):
            ti = len(video_opts['input'])
            output_opts[f'disposition:v:{ti}'] = 'attached_pic'

        # Output path and encoding/mapping options (`-y` was set as a global option above).
        final_cmd.output(output, **output_opts)

        # Preserve visibility of the exact argv ffmpeg will run (same idea as joining final_cmd).
        log.info(f'Execute: \x1b[34m{" ".join(final_cmd.arguments)}\x1b[0m')

        if os.path.exists(output):
            if kwargs.get('overwrite', False):
                log.info(f'Overwriting existing \x1b[1m{output}\x1b[0m!')
            else:
                log.info(f'File \x1b[1m{output}\x1b[0m already exists!')
                return 0

        # I like to see the console output from ffmpeg when it runs.
        final_cmd.on('stderr')(log.debug)

        try:
            final_cmd.execute()
        except FFmpegError as e:
            log.error(f'ffmpeg failed: {e.message}')
            return 1
        vidthen = time.time()
        log.info(f'Done processing \x1b[1m{output}\x1b[0m in {round(vidthen-vidnow,4)} seconds!')
        # Restore tty after ffmpeg progress (interactive sessions).
        if sys.stdout.isatty():
            os.system('stty echo -brkint -imaxbel icanon iexten icrnl')
        then = time.time()
        log.info(f'Completed all media in \x1b[4m{round(then-now, 2)}\x1b[0m seconds!')
    return 0
