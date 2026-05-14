# encoding: utf-8

'''
This module will generate a thumbnail for a video based on the content of the video.
'''

import io
import textwrap

from google import genai
from PIL import Image
from subprocess import Popen

from mkzforge import getLogger, const
log = getLogger(__name__)

TEMPLATE_SVG = '''<svg width="1024" height="1024" viewBox="0 0 1024 1024" background="none" xmlns="http://www.w3.org/2000/svg">
    <style>
        .outlined {
            fill: #04547a;
            stroke: #e9e43c;
            stroke-width: 4;
            stroke-linejoin: round;
            stroke-linecap: round;
            paint-order: stroke fill;
            text-anchor: middle;
            dominant-baseline: middle;
            font-family: DejaVu Sans;
            font-weight: 900;
            font-size: %(fontSize)s
        }
    </style>
%(lines)s
</svg>
'''

TEMPLATE_TEXTLINE = '        <text class="outlined" x="292" y="%(yCoord)s">%(line)s</text>'

IMAGE_PROMPT = """Create an interesting thumbnail for my video.
Here's the text content of the video:

---

%(content)s

---

Take the content and craft an interesting, engaging and simple thumbnail based on the context provided, please.
Do NOT include the TikTok logo nor any username in the resulting image.
Thanks!
"""

CANVAS_W = 585
CANVAS_H = 1024
VISIBLE_H = 800                 # TikTok preview visible height
BORDER = 18                     # border around text block (px)
LEADING_RATIO = 0.20            # line gap as a fraction of font size
AVG_CHAR_ASPECT = 0.65          # average glyph width ~= 0.65 * font size (tune to your font)

SAFE_TOP = (CANVAS_H - VISIBLE_H) // 2  # center the 800px safe area in the 1024px canvas
AVAILABLE_H = VISIBLE_H - 2 * BORDER    # vertical room for the text block inside the safe area
AVAILABLE_W = CANVAS_W - 2 * BORDER

GENAI_CLIENT = None

def getClient() -> genai.Client:
    '''
    The client is garbage collected when it falls out of scope.
    Create an instance and store it here in the module to ensure it keeps the connection alive.
    '''
    global GENAI_CLIENT
    if GENAI_CLIENT == None:
        GENAI_CLIENT = genai.Client(api_key=const.GOOGLE_API_KEY)
    return GENAI_CLIENT

def _wrap_title_12(title: str) -> list[str]:
    """
    Wrap to ~12 chars/line, allowing long words to exceed bounds (no forced breaking).
    Limit to at most 3 lines (merge overflow into the 3rd line).
    """
    lines = textwrap.wrap(
        title.strip(),
        width=12,
        break_long_words=False,     # allow long words to exceed bounds
        break_on_hyphens=False
    )
    if not lines:
        lines = [title.strip()]  # safety, though caller guarantees ≥1 char

    if len(lines) > 3:
        # Merge any overflow into the 3rd line to keep 3 lines max
        lines = lines[:2] + [' '.join(lines[2:])]
    return lines


def _max_font_height_fit(line_count: int) -> int:
    """
    Compute the max font size (px) that fits vertically in AVAILABLE_H
    given a `line_count` and LEADING_RATIO.
    """
    denom = line_count + (line_count - 1) * LEADING_RATIO
    return int(AVAILABLE_H // denom)

def _max_font_width_fit(longest_line_chars: int) -> int:
    n = max(1, longest_line_chars)
    return int(AVAILABLE_W // (n * AVG_CHAR_ASPECT))

def max_font_size_for_char_count_single_line(chars: int) -> int:
    fs_v = _max_font_height_fit(1)
    fs_w = _max_font_width_fit(chars)
    return min(fs_v, fs_w)

def compute_thumbnail_typography(title: str):
    """
    Inputs:
    - title (str): single-line title, 1..36 characters (guaranteed by caller).

    Outputs dict with:
    - fontSize: int, font size in px
    - lineCount: int, 1..3
    - yCoords: Tuple[int, ...], top y of each line inside the 1024px canvas
    - lines: Tuple[str, ...], the wrapped text split into 1..3 lines
    """
    lines = _wrap_title_12(title)
    line_count = len(lines)
    longest = max(len(s) for s in lines)

    fs_v = _max_font_height_fit(line_count)
    fs_w = _max_font_width_fit(longest)
    font_size = int(min(fs_v, fs_w))

    leading_px = int(round(font_size * LEADING_RATIO))
    block_h = line_count * font_size + (line_count - 1) * leading_px
    top_y = SAFE_TOP + BORDER + (AVAILABLE_H - block_h) // 2
    y_coords = tuple(int(top_y + i * (font_size + leading_px) + font_size) for i in range(line_count))

    return {
        "fontSize": font_size,
        "lineCount": line_count,
        "yCoords": y_coords,
        "lines": tuple(lines),
    }

def generate_template(title: str) -> str:
    '''
    Generate the template SVG and write to PNG from the tilte.
    Calculates the position, font size and placement of the title.
    Generates an SVG and writes the result to PNG.
    Return the build artifact to be converted into the live thumbnail.
    '''
    EOL = '\n'
    coverInfo = compute_thumbnail_typography(title)
    log.info(f'Generating SVG->PNG from "{EOL.join(coverInfo["lines"])}"')
    lines = []
    svgVars = {
        'fontSize': coverInfo["fontSize"],
    }
    for i, line in enumerate(coverInfo['lines']):
        cover = {
            'fontSize': coverInfo["fontSize"],
            'lineCount': coverInfo["lineCount"],
            'yCoord': coverInfo["yCoords"][i], # Singular because it's only one coordinate.
            'line': line, # Singular because it's just the 1 line.
        }
        lines.append(TEMPLATE_TEXTLINE % cover)
    svgVars['lines'] = EOL.join(lines)
    svg = TEMPLATE_SVG % svgVars
    log.debug(f'SVG: \n{svg}')
    svg_file_path = 'build/thumbnail.svg'
    open(svg_file_path, 'w').write(svg)
    template = 'build/thumbnail.png'
    # Create the thumbnail from SVG using ImageMagick
    convert_cmd = ['rsvg-convert', svg_file_path, '-o', template]
    log.info(f'Running command: {" ".join(convert_cmd)}')
    p = Popen(convert_cmd)
    p.communicate()

    if p.returncode != 0:
        log.error(f"Error: convert command failed with return code {p.returncode}")
        return ''
    log.info(f'Converted SVG written to {template} as PNG')
    return template

def generate_thumbnail(title: str, content: str) -> str:
    '''
    Take the generated thumbnail template and produce the finished thumbnail.
    '''
    imgpath = generate_template(title)
    log.info('Editing image...')
    src_img = Image.open(imgpath).convert("RGBA")
    log.info('Image loaded.')

    prompt = IMAGE_PROMPT % {'content': content}
    result = getClient().models.generate_content(
        model=const.IMAGE_MODEL_ID,
        contents=[prompt, src_img],
    )

    image_parts = [
        part.inline_data.data
        for part in result.candidates[0].content.parts or [] # type: ignore
        if part.inline_data
    ]

    # Most SDK builds return a PIL.Image with .save available
    thumbnail = 'thumbnail.png'
    if image_parts:
        image = Image.open(io.BytesIO(image_parts[0])) # type: ignore
        image.save(thumbnail)
    return thumbnail
