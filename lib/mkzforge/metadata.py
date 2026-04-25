'''
Handles generation and management of metadata associated with a video.
'''
import os
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from mkzforge import getLogger, const, utils, videos

log = getLogger(__name__)

LLM = None

GENERATE_TITLE_PROMPT = '''Generate a title.
Based on the subtitles provided, summarize the post in a 1-3 word summary that would be an engaging title.
No markdown or extra formatting accepted.
Just the 1 to 3 word summary.
'''

GENERATE_DESCRIPTION_PROMPT = '''Summarize the video content for the video description.
If the video is short (less than 60 seconds), give a one sentence summary.
If multiple points were described, use bullet points to highlight them.
Markdown is not allowed in descriptions.
Keep it under 5000 characters.
Write in first person as if Markizano is speaking directly to the viewer.
'''

def getClient():
    '''
    Get a connection to the LLM endpoint for dynamically generating title and description as necessary.
    '''
    global LLM
    if LLM is None:
        LLM = init_chat_model(
            model=const.LLM_MODEL,
            model_provider=const.LLM_PROVIDER,
        )
    return LLM

def generateMetadata(video_cfg: dict, md_type: Literal['title', 'description'], **kwargs) -> dict:
    '''
    Generate a title|description for a video based on the subtitles/content of the video.
    Reads the text transcript file for the resource and sends its content to the LLM.
    '''
    try:
        resource = video_cfg['input'][0]['i']
        txt_path = f'build/{utils.filename(resource)}.txt'
        if video_cfg['metadata'].get(md_type, '') and not kwargs.get('overwrite', False):
            log.info(f'Video already has {md_type}. Not wasting tokens for another...')
            return video_cfg

        if not os.path.exists(txt_path):
            log.warning(f'Transcript file {txt_path} not found. Cannot generate {md_type}.')
            video_cfg['metadata'][md_type] = ''
            return video_cfg

        log.info(f'Generating {md_type} for \x1b[1m{resource}\x1b[0m from transcript at {txt_path}')

        # Read the transcript file content
        subtitle_content = open(txt_path, 'r', encoding='utf-8').read()

        if md_type == 'title':
            sysprompt = GENERATE_TITLE_PROMPT
        elif md_type == 'description':
            sysprompt = GENERATE_DESCRIPTION_PROMPT
        else:
            # Realistically, this should never execute, but as a preventative measure...
            raise ValueError(f'Unsupported md_type: {md_type}; must be one of "title" or "description".')
        messages = [
            SystemMessage(content=sysprompt),
            HumanMessage(content=subtitle_content)
        ]
        response = getClient().invoke(messages)
        value = str(response.content).strip()
    except Exception as e:
        log.error(f'Exception generating {md_type}: {e}')
        value = ''
    args = {
        md_type: value
    }
    videos.updateVideo(video_cfg, **args)
    return video_cfg
