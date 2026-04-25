'''
CherryPy Web Server for mkzforge

This module provides a web interface for uploading videos and managing mkzforge projects.
It includes:
- HTML page serving
- REST API endpoints for video processing
- Background processing workers
- Project management
'''

import os
import json
import multiprocessing
import cherrypy
from typing import Dict, List, Union
from cherrypy._cpreqbody import Part

from mkzforge import getLogger, notify, utils, videos, subtitles, metadata, genimg, grive

log = getLogger(__name__)
DEBUG = os.getenv('DEBUG', False)

class PageHandlers:
    '''
    Serves HTML pages from webroot directory.
    '''

    def __init__(self, config: dict):
        self.config = config
        self.workspace = config.get('workspace', os.getcwd())
        self.webroot = config.get('webroot', os.path.join(os.getcwd(), 'web'))

    @cherrypy.expose
    def index(self):
        '''Serve video submission form'''
        index_path = os.path.join(self.webroot, 'index.html')
        if not os.path.exists(index_path):
            raise cherrypy.HTTPError(404, 'index.html not found')
        return open(index_path, 'r').read()

    @cherrypy.expose
    def videos(self):
        '''Serve projects list page'''
        projects_path = os.path.join(self.webroot, 'projects.html')
        if not os.path.exists(projects_path):
            raise cherrypy.HTTPError(404, 'projects.html not found')
        return open(projects_path, 'r').read()

    @cherrypy.expose
    def video(self, project=None):
        '''Serve project detail page'''
        if not project:
            raise cherrypy.HTTPError(400, 'Project parameter required')
        detail_path = os.path.join(self.webroot, 'project-detail.html')
        if not os.path.exists(detail_path):
            raise cherrypy.HTTPError(404, 'project-detail.html not found')
        return open(detail_path, 'r').read()

    @cherrypy.expose
    def oauth2callback(self, **kwargs):
        '''
        GET /oauth2callback?state=...&code=...&scope=...
        Receives the full Google OAuth2 redirect URL, exchanges it for a token, then
        redirects home.  The complete query string is passed to grive so oauthlib can
        validate state, scope, and all other parameters Google includes in the response.
        '''
        if not kwargs.get('code'):
            raise cherrypy.HTTPError(400, 'Missing authorization code')
        http_host = self.config.get('http_host', f"http://{cherrypy.request.headers.get('Host', 'localhost')}")
        redirect_uri = f'{http_host}/oauth2callback'
        # Reconstruct the full authorization response URL that Google redirected to.
        qs = cherrypy.request.query_string
        authorization_response = f'{redirect_uri}?{qs}'
        grive.handle_oauth_callback(self.config, authorization_response, redirect_uri)
        raise cherrypy.HTTPRedirect('/')


class GriveHandlers:
    '''
    Sub-handler for /api/grive/* endpoints.
    Mounted as ApiHandlers.grive so CherryPy routes /api/grive/* here automatically.
    '''

    def __init__(self, config: dict):
        self.config = config

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def status(self):
        '''GET /api/grive/status — returns {"authenticated": bool}'''
        log.info('GET /api/grive/status')
        return {'authenticated': grive.is_authenticated()}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def list(self):
        '''GET /api/grive/list — returns {"files": [...]} or 401 if not authenticated'''
        log.info('GET /api/grive/list')
        if not grive.is_authenticated():
            cherrypy.response.status = 401
            return {'error': 'not_authenticated'}
        try:
            files = grive.list_folder(self.config)
            return {'files': files}
        except Exception as e:
            log.error(f'Failed to list Drive folder: {e}', exc_info=True)
            raise cherrypy.HTTPError(500, str(e))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def auth(self):
        '''GET /api/grive/auth — returns {"auth_url": "..."}'''
        log.info('GET /api/grive/auth')
        http_host = self.config.get('http_host',
            f"http://{cherrypy.request.headers.get('Host', 'localhost')}")
        redirect_uri = f'{http_host}/oauth2callback'
        auth_url = grive.get_auth_url(self.config, redirect_uri)
        return {'auth_url': auth_url}


class ApiHandlers:
    '''
    JSON API endpoints for video processing and project management.
    '''

    def __init__(self, config: dict):
        self.config = config
        self.workspace = config.get('workspace', os.getcwd())
        self.processing_jobs: Dict[str, dict] = {}
        self.grive = GriveHandlers(config)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def projects(self):
        '''
        GET /api/projects
        Returns list of all projects in workspace
        '''
        log.info(f'{cherrypy.request.method} /api/projects.')
        if cherrypy.request.method != 'GET':
            log.error(f'Method {cherrypy.request.method} not allowed. Use GET only.')
            raise cherrypy.HTTPError(405, 'Method not allowed')
        try:
            projects = []
            if not os.path.exists(self.workspace):
                return {'projects': projects}
            for project_name in os.listdir(self.workspace):
                project_path = os.path.join(self.workspace, project_name)
                if not os.path.isdir(project_path):
                    continue
                # Just check right quick if it's a `mkzforge.yml` configured folder.
                config_path = os.path.join(project_path, 'mkzforge.yml')
                if not os.path.exists(config_path): continue
                projects.append(project_name)
            return {'projects': projects}

        except Exception as e:
            log.error(f'Failed to list projects: {e}')
            raise cherrypy.HTTPError(500, str(e))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def project(self, project_name):
        '''
        GET /api/project/<project_name>
        Returns detailed information about a specific project
        '''
        log.info(f'{cherrypy.request.method} /api/project/{project_name}.')
        if cherrypy.request.method != 'GET':
            raise cherrypy.HTTPError(405, 'Method not allowed')

        try:
            project_path = os.path.join(self.workspace, project_name)
            if not os.path.exists(project_path):
                raise cherrypy.HTTPError(404, f'Project {project_name} not found')

            # Read configuration
            project_config = None
            if os.path.exists(os.path.join(project_path, 'mkzforge.yml')):
                project_config = utils.load(project_path)
            return {
                'name': project_name,
                'config': project_config,
            }

        except Exception as e:
            log.error(f'Failed to get project details: {e}')
            raise cherrypy.HTTPError(500, str(e))

    @cherrypy.expose
    def resource(self, project_name: str, resource: str):
        '''
        GET /api/resource/<project_name>/(thumbnail|output)
        Returns a resource from the project.
        '''
        log.info(f'{cherrypy.request.method} /api/project/{project_name}/{resource}.')
        if cherrypy.request.method != 'GET':
            raise cherrypy.HTTPError(405, 'Method not allowed')

        try:
            project_path = os.path.join(self.workspace, project_name)
            if not os.path.exists(project_path):
                raise cherrypy.HTTPError(404, f'Project {project_name} not found')

            # Read configuration
            project_config = None
            if os.path.exists(os.path.join(project_path, 'mkzforge.yml')):
                project_config = utils.load(project_path)
            if resource == 'output':
                output_filename = os.path.join(self.workspace, project_name, project_config['videos'][-1]['output'])
                attachment_filename = os.path.basename(output_filename)
                cherrypy.response.headers['Content-Type'] = 'video/mp4' if attachment_filename.endswith('mp4') else 'video/x-matroska'
            if resource == 'thumbnail':
                output_filename = os.path.join(self.workspace, project_name, 'thumbnail.png')
                attachment_filename = f'{project_name}.png'
                cherrypy.response.headers['Content-Type'] = 'image/png'
            if not os.path.exists(output_filename):
                raise cherrypy.HTTPError(404, f'Resource {resource} not found')
            cherrypy.response.headers['Content-Disposition'] = f'attachment; filename="{attachment_filename}"'
            return open(output_filename, 'rb').read()

        except Exception as e:
            log.error(f'Failed to get project details: {e}')
            raise cherrypy.HTTPError(500, str(e))

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def process(self, project_config: str, video_inputs: Union[Part, List[Part], None] = None,
                grive_files: str = None, grive_names: str = None):
        '''
        POST /api/process
        Upload videos and start processing pipeline.
        Accepts either uploaded video parts or grive_files/grive_names JSON lists.
        '''
        log.info(f'{cherrypy.request.method} /api/process.')
        if cherrypy.request.method != 'POST':
            raise cherrypy.HTTPError(405, 'Method not allowed')
        log.info('Receiving upload.')
        # mkzforge.cli.new.gennew will cd into the project directory. On each request, we need to revert that.
        os.chdir(self.workspace)

        try:
            project_cfg = json.loads(project_config)
            # Google Drive path: download files then hand off to pipeline
            if grive_files:
                file_ids = json.loads(grive_files)
                file_names = json.loads(grive_names) if grive_names else []
                if file_ids:
                    if not project_cfg.get('name'):
                        raise cherrypy.HTTPError(400, 'project_name is required')
                    videos.newProject(resource=project_cfg['name'], **self.config)
                    for file_id, file_name in zip(file_ids, file_names):
                        destination = os.path.join(
                            self.workspace, project_cfg['name'], 'resources', file_name
                        )
                        grive.download_file(file_id, destination)
                    if DEBUG:
                        process_video_pipeline(self.config, project_cfg)
                    else:
                        proc = multiprocessing.Process(
                            target=process_video_pipeline,
                            args=(self.config, project_cfg),
                            daemon=True
                        )
                        proc.start()
                    log.info(f'Started background processing for project: {project_cfg["name"]}')
                    return {
                        'status': 'success',
                        'message': 'Drive files queued. Processing started in background.',
                        'project': project_cfg['name']
                    }

            if video_inputs is None:
                video_list = []
            elif not isinstance(video_inputs, list):
                video_list = [video_inputs]
            else:
                video_list = video_inputs

            # Validation
            if not project_cfg['name']:
                raise cherrypy.HTTPError(400, 'project_name is required')
            if not video_inputs:
                raise cherrypy.HTTPError(400, 'At least one video file is required')

            videos.newProject(resource=project_cfg['name'], **self.config)
            log.info(f'Got JSON project config: {project_cfg}')
            project_cfg = json.loads(project_cfg)

            for i, video in enumerate(video_list):
                log.info(f'Received video {video.filename} uploading...')
                if project_cfg['videos'] and project_cfg['videos'][0]['input']:
                    video_filename: str = os.path.basename(project_cfg['videos'][0]['input'][i]['i'])
                else:
                    video_filename: str = os.path.basename(video.filename)
                video_path = os.path.join(self.workspace, project_cfg['name'], 'resources', video_filename)
                # Chunk the video because it could be many GB in size. We don't want that in-memory.
                with open(video_path, 'wb') as fd:
                    while True:
                        chunk = video.file.read(8192)
                        if not chunk:
                            break
                        fd.write(chunk)
                        fd.flush()
                log.info(f'Saved uploaded video: {video_path} ({os.path.getsize(video_path)} bytes).')

                if DEBUG:
                    # Stay attached in debug mode because I may do some interactive testing.
                    process_video_pipeline(self.config, project_cfg['name'], project_cfg)
                else:
                    # Start background processing
                    process = multiprocessing.Process(
                        target=process_video_pipeline,
                        args=(self.config, project_cfg['name'], project_cfg),
                        daemon=True
                    )
                    process.start()

                log.info(f'Started background processing for project: {project_cfg["name"]}')

            return {
                'status': 'success',
                'message': 'Video upload successful. Processing started in background.',
                'project': project_cfg['name']
            }
        except cherrypy.HTTPError:
            raise
        except Exception as e:
            log.error(f'Failed to process upload: {e}', exc_info=True)
            raise cherrypy.HTTPError(500, str(e))

def process_video_pipeline(cfg: dict, project_config: dict):
    '''
    Background worker for video processing.
    Uses built-in mkzforge interfaces instead of shell commands.
    '''

    log.info(f'Starting video pipeline for project: {project_config["name"]}')
    project_path = os.path.join(cfg['workspace'], project_config["name"])
    os.chdir(project_path)
    log.info(f'Process config: {project_config}')

    try:
        log.info('Normalizing resources from existing video list.')
        # I'm aware this looks a lot like `mkzforge.cli.normalize:normalize`.
        # The challenge is the circular import if I pull the CLI function in here.
        # It's already simplified to just these interfaces, so this is OK, IMHO.
        mkzforge_cfg = utils.load()
        cfg['name'] = os.path.basename(os.getcwd())
        video_cfg, resource = videos.detectState(mkzforge_cfg, **cfg)
        subtitles.genSubtitles(video_cfg, resource, **cfg)
        # Inject title and description into the mix before attempting to generate them.
        for descriptor in ['title', 'description']:
            if descriptor in project_config and project_config[descriptor]:
                video_cfg['metadata'][descriptor] = project_config[descriptor]
        metadata.generateMetadata(video_cfg, 'title', **cfg)
        metadata.generateMetadata(video_cfg, 'description', **cfg)
        # Set filter_complex to None to get the default hardsub filter.
        videos.updateVideo(video_cfg, attributes=['thumbnail'], filter_complex=None)
        content = open(f'build/{utils.filename(resource)}.txt').read()
        if not os.path.exists('thumbnail.png') or ( os.path.exists('thumbnail.png') and cfg.get('overwrite', False) ):
            genimg.generate_thumbnail(video_cfg['metadata']['title'], content)

        if not utils.hasInput(mkzforge_cfg['videos'], resource):
            mkzforge_cfg['videos'].append(video_cfg)
        utils.save(mkzforge_cfg['videos'])
        log.info('Video(s) normalized and added to `mkzforge.yml` config.')

        log.info(f'In-memory project config: {project_config}')
        log.info(f'In-memory MKZ config: {mkzforge_cfg}')

        log.info('Building compiled final video result.')
        log.debug(f'Build config: {mkzforge_cfg}')
        videos.compileVideo(video_cfg, **cfg)

        # Send success notification
        if not DEBUG:
            notify.send_notification(
                'INFO',
                f'mkzforge: {project_config["name"]} complete',
                f'Video processing completed successfully for project: {project_config["name"]}'
            )

        log.info(f'Video pipeline completed successfully for: {project_config["name"]}')

    except Exception as e:
        log.error(f'Video pipeline failed for {project_config["name"]}: {e}', exc_info=True)

        # Send error notification
        if not DEBUG:
            notify.send_notification(
                'ERROR',
                f'mkzforge: {project_config["name"]} failed',
                f'Video processing failed for project {project_config["name"]}: {str(e)}'
            )

def jsonify_error(status, message, traceback, version):
    '''
    Custom error handler to return JSON instead of HTML.
    '''
    cherrypy.response.headers['Content-Type'] = 'application/json'

    # Log the traceback to the console manually if needed
    log.error(f"Error {status}: {message}\n{traceback}")

    return json.dumps({
        'status': status,
        'message': message,
        'version': version
    }).encode('utf-8')
