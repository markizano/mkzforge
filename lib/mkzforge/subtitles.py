'''
All functions and process related to subtitles.
This includes subtitle generation and translation.
General flow: Input file -> Get back subtitles.
'''
import os
import re
import time
import subprocess
from mkzforge import getLogger, types, utils, videos
log = getLogger(__name__)

# I know this isn't an end-all solution, but it handles 80% of the cases Whisper gets it wrong.
markizano = re.compile(r'm[ae]r\w*[ao]no', re.I)
kizano = re.compile(r'\bk[iu][sz][ao]n[oa]', re.I)
draconus = re.compile(r'dr[au]c[ao]nis', re.I)
tanninovian = re.compile(r't[ae]nn?[aie]n?n?o[bv]?i?[ae]n', re.I)
# Caligrapher's marks
fsck = re.compile(r'fuck', re.I)

def fixSubtitles(srt_path: str) -> str:
    '''
    Whisper is constantly mis-spelling my name and various other story-lore.
    I attempt to correct that here.
    '''
    log.info(f'Implementing corrections to {srt_path}')
    subtitles = open(srt_path).read()
    subtitles = kizano.sub('Kizano', subtitles)
    subtitles = markizano.sub('Markizano', subtitles)
    subtitles = draconus.sub('Draconus', subtitles)
    subtitles = tanninovian.sub('Tanninovian', subtitles)
    subtitles = fsck.sub('fsck', subtitles)
    open(srt_path, 'w').write(subtitles)
    return subtitles

def whisperModel() -> str:
    '''
    Automatically select the best Whisper model based on available GPU VRAM.

    Model VRAM requirements (approximate):
    - tiny:     ~1 GB  VRAM
    - base:     ~1 GB  VRAM
    - small:    ~2 GB  VRAM
    - medium:   ~5 GB  VRAM
    - large:    ~10 GB VRAM
    - large-v2: ~10 GB VRAM
    - large-v3: ~10 GB VRAM

    Returns the model name to use with Whisper.
    '''
    # Auto-select based on GPU VRAM
    vram_mb = utils.get_gpu_vram_mb()

    if vram_mb == 0:
        # No GPU, use small model for CPU
        log.info('No GPU detected, selecting "small" model for CPU processing')
        return 'small'
    elif vram_mb >= 10240:  # 10 GB or more
        log.info(f'GPU has {vram_mb} MB VRAM, selecting "large-v3" model (best quality)')
        return 'large-v3'
    elif vram_mb >= 8192:   # 8-10 GB
        log.info(f'GPU has {vram_mb} MB VRAM, selecting "large-v2" model (excellent quality)')
        return 'large-v2'
    elif vram_mb >= 6144:   # 6-8 GB
        log.info(f'GPU has {vram_mb} MB VRAM, selecting "medium" model (good quality)')
        return 'medium'
    elif vram_mb >= 3072:   # 3-6 GB
        log.info(f'GPU has {vram_mb} MB VRAM, selecting "small" model (decent quality)')
        return 'small'
    else:                    # < 3 GB
        log.info(f'GPU has {vram_mb} MB VRAM, selecting "base" model (basic quality)')
        return 'base'

def genSubtitles(
    video_cfg: dict,
    video_path: str,
    **kwargs
) -> dict:
    '''
    Generate subtitles for a video file using the whisper script directly.
    Returns the path to the generated SRT file.
    '''
    lang = utils.language(kwargs, video_cfg)
    srtfile = os.path.join('build', f"{utils.filename(video_path)}.{lang}.srt")
    txtfile = os.path.join('build', f"{utils.filename(video_path)}.txt")
    log.info(f"Generating subtitles for {srtfile} from {video_path}... This might take a while...")

    if os.path.exists(srtfile):
        if kwargs.get('overwrite', False):
            log.info(f"Overwriting existing subtitles for \x1b[1m{srtfile}\x1b[0m!")
        else:
            log.info(f"Subtitles already generated for \x1b[1m{srtfile}\x1b[0m!")
            log.info('Checking to see if sub config is in mkzforge.yml...')
            for inputVid in video_cfg['input']:
                if inputVid['i'].endswith('.srt'):
                    log.info('Config is good!')
                    return video_cfg
            subs = [(lang, srtfile)]
            log.info(f'Video config missing SRT files! Adding subs={subs}')
            videos.updateVideo(video_cfg, subs=subs)
            return video_cfg

    with utils.video_processing_lock('subtitles'):
        # Build whisper command
        log.info(f'PATH: {os.getenv("PATH")}')
        vram_mb = utils.get_gpu_vram_mb()
        whisper_cmd = [
            'whisper',
            # Select appropriate Whisper model based on GPU VRAM
            '--model', whisperModel(),
        ]
        # Select the appropriate device based on GPU presence.
        if vram_mb == 0:
            whisper_cmd.extend(['--devices', types.Devices.CPU])
        else:
            whisper_cmd.extend([
                '--device', types.Devices.CUDA,
                '--fp16', 'False',
            ])
        whisper_cmd.extend([
            '--output_dir', 'build',
            '--output_format', 'all',
            '--language', lang,
            '--task', kwargs.get('task', types.WhisperTask.TRANSCRIBE),
            '--word_timestamps', 'True',
            '--max_words_per_line', '5',
            '--highlight_words', 'True',
            '--verbose', 'True',
            '--initial_prompt', (
                'Markizano Draconus is a Tanninovian from the Crux galaxy. '
                "Kizano's FinTech is an education and career advancement company. "
                'markizano.net is the website you can visit. '
                'Alex Hormozi and Codie Sanchez are YouTube personalities.'
            )
        ])

        # Add the video file
        whisper_cmd.append(video_path)

        log.info(f"Running whisper command: {' '.join(whisper_cmd)}")
        try:
            now = time.time()
            # Use Popen to stream output to the logger in real-time
            process = subprocess.Popen(
                whisper_cmd,
                text=True,
                env=os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip('\n')
                if line:
                    log.info(line)
            returncode = process.wait()
            then = time.time()

            if returncode != 0:
                log.error(f"Whisper failed with exit code {returncode}")
                return video_cfg

            log.info(f"Whisper completed in {round(then-now, 4)} seconds!")
            log.info(f'Now removing excess VTT, JSON, and TSV files...')
            for suffix in ['json', 'vtt', 'tsv']:
                extra = os.path.join('build', f'{utils.filename(video_path)}.{suffix}')
                if os.path.exists(extra):
                    os.unlink(extra)

            # Whisper will create the SRT file with the same name as the video but with .srt extension
            # We need to rename it to match our expected naming convention
            generated_whisper_srt = os.path.join('build', f"{utils.filename(video_path)}.srt")
            if os.path.exists(generated_whisper_srt) and generated_whisper_srt != srtfile:
                os.rename(generated_whisper_srt, srtfile)
                log.info(f"Renamed {generated_whisper_srt} to {srtfile}")

            fixSubtitles(srtfile)
            fixSubtitles(txtfile)
            videos.updateVideo(video_cfg, subs=[(lang, srtfile)])

            return video_cfg
        except Exception as e:
            log.error(f"Failed to generate subtitles: {e}")
            return video_cfg
