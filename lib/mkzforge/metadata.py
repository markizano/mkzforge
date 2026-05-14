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

GENERATE_DESCRIPTION_PROMPT = '''You are a video metadata assistant.
The user will provide a raw video transcript.
Write a single-paragraph description (3-5 sentences) of what the video covers.
Write in first person as if Markizano is speaking directly to the viewer.
Keep it under 5000 characters.
Plain text only. No markdown. No lists.'''

GENERATE_TITLE_PROMPT = '''You are a video metadata assistant.
The user will provide a short video description.
Return ONLY a title: 2-4 words, plain text, no punctuation, no markdown.
Example input: "This video walks through setting up an ECS cluster with Terraform..."
Example output: ECS Terraform Setup'''

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
        content = open(txt_path, 'r', encoding='utf-8').read()

        if md_type == 'title':
            sysprompt = GENERATE_TITLE_PROMPT
            # If we're generating the title, try to do it from the description.
            # I've had long-form video where the title ends up being a huge summary of the video
            # no matter how strict I am with the prompt.
            content = video_cfg['metadata'].get('description', content)
        elif md_type == 'description':
            sysprompt = GENERATE_DESCRIPTION_PROMPT
        else:
            # Realistically, this should never execute, but as a preventative measure...
            raise ValueError(f'Unsupported md_type: {md_type}; must be one of "title" or "description".')
        messages = [
            SystemMessage(content=sysprompt),
            HumanMessage(content=content)
        ]
        response = getClient().invoke(messages)
        value = str(response.content).strip()
        if md_type == 'title' and  len(value.split()) > 5:
            log.warning(f'TITLE longer than 5 words, re-generating: {value}')
            value = generateMetadata(video_cfg, 'title', **kwargs)
    except Exception as e:
        log.error(f'Exception generating {md_type}: {e}')
        value = ''
    args = {
        md_type: value
    }
    videos.updateVideo(video_cfg, **args)
    return video_cfg
