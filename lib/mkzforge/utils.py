'''
Ancilary utility functions to help processing.
'''
import os
import time
import subprocess
import fcntl
import random
from contextlib import contextmanager
from glob import glob
from kizano.utils import read_yaml, write_yaml

from mkzforge import getLogger, notify
log = getLogger(__name__)

_gpu_vram_mb = None

def language(cfg: dict, video_cfg: dict) -> str:
    '''
    Try to parse out the language from the config/env.
    Definitely return a language. Default to English if nothing else.
    '''
    lang_env = os.getenv('LANGUAGE', 'en')
    lang_cfg = cfg.get('language', lang_env)
    lang_vid = video_cfg.get('language', lang_cfg)
    return lang_vid

def filename(path: str) -> str:
    '''
    Gets the filename without the extension or leading path.
    '''
    return os.path.splitext(os.path.basename(path))[0]

def hasInput(videos: list[dict], artifact: str) -> bool:
    '''
    Check the list of videos.
    If the target artifact (subtitle|video) is mentioned in the inputs, return True
    '''
    for video in videos:
        for inputVid in video['input']:
            if inputVid['i'] == artifact:
                return True
    return False

def hasConcat(videos: list[dict]) -> bool:
    '''
    Does this config set have a `concat` operation?
    '''
    for video in videos:
        if 'filter_complex' not in video: continue
        for filter_complex in video['filter_complex']:
            if 'concat' in filter_complex:
                return True
    return False

def getInputIndex(videos: list[dict], artifact: str) -> int|None:
    '''
    Check the list of videos.
    Return the video index the input was found so we can extract the `video_cfg` for it.
    '''
    for i, video in enumerate(videos):
        for inputVid in video['input']:
            if inputVid['i'] == artifact:
                return i
    return None

def load(path_prefix: str = '.') -> dict:
    '''
    Read `mkzforge.yml` config into memory.
    '''
    cfg_path = os.path.abspath(os.path.join(path_prefix, 'mkzforge.yml'))
    log.info(f'Reading {cfg_path} config...')
    return read_yaml(cfg_path)

def save(videos: list[dict]) -> None:
    '''
    Write mkzforge.yml updates to disk.
    '''
    log.info('Writing out mkzforge.yml configuration...')
    write_yaml('mkzforge.yml', { 'videos': videos })
    log.info('Done writing out mkzforge.yml configuration!')

def getMP4s() -> list[str]:
    '''
    List all MP4 files in the resources folder.
    '''
    return glob('resources/*.mp4')

def getResources() -> list[str]:
    '''
    List all resources in the resources folder.
    '''
    return glob('resources/*.mp4') + glob('resources/*.mkv')

def getTranscripts() -> list[str]:
    '''
    List all whisper transcripts.
    These are the TXT files in the `build/` folder.
    '''
    return glob('build/*.txt')

def mergefilters(segments: list) -> list[str]:
    '''
    Generate and return the final `concat` filter with all the segments properly input.
    required to converge them together.

    segments: An array with as many elements that are meant to be merged.

    Returns: filter complex array that can be `.extend()`ed to define the concat.
    Outputs to [video] and [audio] streams to map.
    '''
    trim_filters = []
    if len(segments) > 1:
        video_inputs = ''.join([f'[v{i}]' for i in range(len(segments))])
        audio_inputs = ''.join([f'[a{i}]' for i in range(len(segments))])
        concat_filter = f"{video_inputs}concat=n={len(segments)}:v=1:a=0,setsar=1:1[video];{audio_inputs}concat=n={len(segments)}:v=0:a=1[audio]"
        trim_filters.append(concat_filter)
    else:
        # Single segment, just rename outputs
        trim_filters.append("[v0]null[video]")
        trim_filters.append("[a0]anull[audio]")

    return trim_filters

def get_gpu_vram_mb() -> int:
    '''
    Detect available GPU VRAM in megabytes.
    Returns 0 if no GPU detected or on error.
    '''
    global _gpu_vram_mb
    if _gpu_vram_mb is not None:
        return _gpu_vram_mb

    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Get first GPU's VRAM (in MB)
            vram_mb = int(result.stdout.strip().split('\n')[0])
            _gpu_vram_mb = vram_mb
            log.info(f'Detected GPU with {vram_mb} MB VRAM via nvidia-smi')
            return vram_mb
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError) as e:
        log.debug(f'nvidia-smi detection failed: {e}')

    # Try torch if available
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_mb = int(vram_bytes / (1024 * 1024))
            _gpu_vram_mb = vram_mb
            log.info(f'Detected GPU with {vram_mb} MB VRAM via torch')
            return vram_mb
    except (ImportError, RuntimeError) as e:
        log.debug(f'torch detection failed: {e}')

    # No GPU detected
    log.warning('No GPU detected, will use CPU or smaller models')
    _gpu_vram_mb = 0
    return 0

@contextmanager
def video_processing_lock(operation: str, lockfile_timeout: int = 3600):
    '''
    Context manager for acquiring an exclusive lock for video processing operations.

    Prevents multiple refresh or build processes from running simultaneously.
    Uses POSIX file locking (fcntl.flock) with retry logic and random delays to avoid race conditions.

    Args:
        operation: The operation name (e.g., 'refresh', 'build') for logging purposes

    Usage:
        with self.video_processing_lock('refresh'):
            # Run refresh operation
            self.process_videos()
    '''
    lockfile = os.path.expanduser('~/.mkzforge.lock')
    lock_file = None
    lock_acquired = False
    start_time = time.time()

    try:
        # Create lock file if it doesn't exist
        lock_dir = os.path.dirname(lockfile)
        if lock_dir and not os.path.exists(lock_dir):
            try:
                os.makedirs(lock_dir, mode=0o755, exist_ok=True)
            except (OSError, PermissionError) as e:
                log.error(f'Failed to create lockfile directory {lock_dir}: {e}')
                raise

        # Open/create lock file
        lock_file = open(lockfile, 'w')

        # Try to acquire lock with retry logic
        while True:
            try:
                # Try non-blocking lock first
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True

                # Write current PID to lockfile
                lock_file.write(str(os.getpid()))
                lock_file.flush()

                log.info(f'Acquired video processing lock for {operation}: {lockfile} (PID: {os.getpid()})')
                break
            except IOError:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed >= lockfile_timeout:
                    error_msg = (
                        f'Failed to acquire video processing lock for {operation} after {elapsed:.1f}s '
                        f'(timeout: {lockfile_timeout}s). '
                        f'Another mkzforge process may be stuck or still running. '
                        f'Check lockfile: {lockfile}'
                    )
                    log.error(error_msg)

                    # Send ERROR notification
                    try:
                        notify.send_notification(
                            'ERROR',
                            f'mkzforge {operation} lock timeout',
                            error_msg
                        )
                    except Exception as notify_error:
                        log.warning(f'Failed to send notification: {notify_error}')

                    raise RuntimeError(error_msg)

                # Wait 1 second + random 0.1-1.0s to mitigate race conditions
                wait_time = 1.0 + random.uniform(0.1, 1.0)
                log.warning(
                    f'Video processing lock for {operation} is held by another process. '
                    f'Waiting {wait_time:.2f}s... (elapsed: {elapsed:.1f}s/{lockfile_timeout}s)'
                )
                time.sleep(wait_time)

        # Yield control to caller (lock is held)
        yield

    finally:
        # Always release lock and close file
        if lock_file:
            if lock_acquired:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    log.info(f'Released video processing lock for {operation}: {lockfile}')
                except Exception as e:
                    log.warning(f'Error releasing video processing lock: {e}')

            try:
                lock_file.truncate(0)
                lock_file.close()
            except Exception as e:
                log.warning(f'Error closing lock file: {e}')
