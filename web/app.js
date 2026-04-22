$(document).ready(function() {
  let videoInputCount = 1;

  // Tab switching
  $(document).on('click', '.tab-btn', function() {
    const tab = $(this).data('tab');
    $('.tab-btn').removeClass('active');
    $('.tab-panel').removeClass('active');
    $(this).addClass('active');
    $(`#tab-${tab}`).addClass('active');
    if (tab === 'gdrive') {
      loadDriveStatus();
    }
  });

  function loadDriveStatus() {
    $.get('/api/grive/status', function(data) {
      if (data.authenticated) {
        loadDriveFiles();
      } else {
        $('#gdrive-content').html(
          '<p>Connect your Google Drive to select videos already uploaded there.</p>' +
          '<button type="button" id="gdrive-connect" class="btn-primary">Connect Google Drive</button>'
        );
        $('#gdrive-connect').on('click', function() {
          $.get('/api/grive/auth', function(res) {
            window.location.href = res.auth_url;
          }).fail(function() {
            showStatus('Failed to get Google Drive auth URL.', 'error');
          });
        });
      }
    }).fail(function() {
      $('#gdrive-content').html('<p class="error">Could not reach server to check Drive status.</p>');
    });
  }

  function loadDriveFiles() {
    $('#gdrive-content').html('<p class="loading">Loading files from Google Drive...</p>');
    $.get('/api/grive/list', function(data) {
      if (!data.files || data.files.length === 0) {
        $('#gdrive-content').html('<p class="info">No video files found in the configured Drive folder.</p>');
        return;
      }
      let html = '<div id="gdrive-file-list">';
      data.files.forEach(function(f) {
        const sizeMB = f.size ? (parseInt(f.size, 10) / (1024 * 1024)).toFixed(1) + ' MB' : 'unknown size';
        // Format modified time in ISO8601 format in browser's timezone
        let modifiedTime = 'unknown date';
        if (f.modifiedTime) {
          const date = new Date(f.modifiedTime);
          // Format as ISO8601 in local timezone (YYYY-MM-DDTHH:mm:ss±HH:mm)
          const year = date.getFullYear();
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const day = String(date.getDate()).padStart(2, '0');
          const hours = String(date.getHours()).padStart(2, '0');
          const minutes = String(date.getMinutes()).padStart(2, '0');
          const seconds = String(date.getSeconds()).padStart(2, '0');
          const offset = -date.getTimezoneOffset();
          const offsetHours = String(Math.floor(Math.abs(offset) / 60)).padStart(2, '0');
          const offsetMinutes = String(Math.abs(offset) % 60).padStart(2, '0');
          const offsetSign = offset >= 0 ? '+' : '-';
          modifiedTime = `${year}-${month}-${day}T${hours}:${minutes}:${seconds}${offsetSign}${offsetHours}:${offsetMinutes}`;
        }
        html += `<div class="gdrive-file-item">
          <input type="checkbox" class="gdrive-check" data-id="${f.id}" data-name="${f.name}">
          <span>${f.name} <small>(${sizeMB}, ${modifiedTime})</small></span>
        </div>`;
      });
      html += '</div>';
      $('#gdrive-content').html(html);
    }).fail(function(xhr) {
      if (xhr.status === 401) {
        $('#gdrive-content').html(
          '<p>Session expired. <button type="button" id="gdrive-connect" class="btn-primary">Reconnect Google Drive</button></p>'
        );
        $('#gdrive-connect').on('click', function() {
          $.get('/api/grive/auth', function(res) {
            window.location.href = res.auth_url;
          });
        });
      } else {
        $('#gdrive-content').html('<p class="error">Failed to load Drive file list.</p>');
      }
    });
  }

  // Add video input handler
  $('#add-video').on('click', function() {
    videoInputCount++;
    const videoInput = `
      <div class="video-input">
        <div class="form-group">
          <label>Video File ${videoInputCount}</label>
          <input type="file" name="video" accept="video/*">
        </div>
        <div class="form-group">
          <label>Override Filename (optional)</label>
          <input type="text" placeholder="custom-name.mp4">
        </div>
      </div>
    `;
    $('#video-inputs').append(videoInput);
  });

  // Form submission handler
  $('#upload-form').on('submit', function(e) {
    e.preventDefault();
    submitForm();
  });

  function submitForm() {
    try {
      const formData = new FormData();

      // Build project configuration matching schema.json
      const projectName = $('#project-name').val().trim();
      const title = $('#title').val().trim();
      const description = $('#description').val().trim();
      const subtitles = $('#subtitles').is(':checked');
      const cut_silence = $('#cut-silence').is(':checked');

      const isDriveTab = $('#tab-gdrive').hasClass('active');

      if (isDriveTab) {
        // Google Drive path: collect selected file IDs and names
        const selectedIds = [];
        const selectedNames = [];
        $('.gdrive-check:checked').each(function() {
          selectedIds.push($(this).data('id'));
          selectedNames.push($(this).data('name'));
        });
        if (selectedIds.length === 0) {
          showStatus('Please select at least one file from Google Drive.', 'error');
          return;
        }
        formData.append('grive_files', JSON.stringify(selectedIds));
        formData.append('grive_names', JSON.stringify(selectedNames));
      } else {
        // Direct upload path
        let hasVideo = false;
        $('.video-input').map(function() {
          const fileInput = $(this).find('input[type="file"]')[0];
          const overrideName = $(this).find('input[type="text"]').val().trim();
          if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const filename = overrideName || file.name;
            formData.append('videos', file, filename);
            hasVideo = true;
            return {video: file, filename};
          }
        });
        if (!hasVideo) {
          showStatus('Please select at least one video file to upload.', 'error');
          return;
        }
      }

      // Project configuration
      const projectConfig = {
        name: projectName,
        subtitles,
        cut_silence,
        title,
        description,
      };
      console.log('Project config payload: ', projectConfig);

      // Add project configuration
      formData.append('project_config', JSON.stringify(projectConfig));

      // Show progress container
      $('#progress-container').show();
      $('#submit-btn').prop('disabled', true);
      hideStatus();

      // AJAX upload with progress tracking
      $.ajax({
        url: '/api/process',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        timeout: 300000, // 5 minutes
        xhr: function() {
          const xhr = new window.XMLHttpRequest();
          xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
              const percent = Math.round((e.loaded / e.total) * 100);
              updateProgress(percent);
            }
          }, false);
          return xhr;
        },
        success: function(response) {
          updateProgress(100);
          showStatus(
            'Upload successful! Your video is being processed. ' +
            'You will receive a notification when complete.',
            'success'
          );

          // Reset form after 3 seconds and redirect to projects
          setTimeout(function() {
            // $('#upload-form')[0].reset();
            $('#submit-btn').prop('disabled', false);
            $('#progress-container').hide();
            // window.location.href = '/videos';
          }, 3000);
        },
        error: function(xhr, textStatus, errorThrown) {
          console.error('Upload error', textStatus, errorThrown, xhr);
          let errorMsg;

          // Network-level errors or server not reachable
          if (xhr.readyState === 0 && xhr.status === 0) {
            errorMsg = 'Cannot connect to server. Please ensure the mkzforge server is running and try again.';
          } else if (textStatus === 'timeout') {
            errorMsg = 'Upload timed out. Please check your connection and server status, then try again.';
          } else if (xhr.responseJSON && xhr.responseJSON.error) {
            errorMsg = xhr.responseJSON.error;
          } else if (xhr.responseJSON && xhr.responseJSON.message) {
            errorMsg = xhr.responseJSON.message;
          } else if (xhr.response && xhr.response.error) {
            errorMsg = xhr.response.error;
          } else if (xhr.responseText) {
            // Try to extract something useful from plain text / HTML
            errorMsg = xhr.responseText.substring(0, 200);
          } else if (errorThrown) {
            errorMsg = errorThrown.toString();
          } else {
            errorMsg = 'Upload failed. Please try again.';
          }

          showStatus('Error: ' + errorMsg, 'error');
          $('#submit-btn').prop('disabled', false);
          $('#progress-container').hide();
        }
      });

    } catch (e) {
      showStatus(`Cannot process form: ${e}`, 'error');
    }
  }

  function updateProgress(percent) {
    $('#progress-fill').css('width', percent + '%');
    $('#progress-text').text('Uploading: ' + percent + '%');
    const position = $('#progress-container').offset();
    window.scrollTo(position.left, position.top)
  }

  function showStatus(message, type) {
    const statusDiv = $('#status-message');
    statusDiv.text(message);
    statusDiv.removeClass('success error');
    statusDiv.addClass(type);
    statusDiv.show();
    const position = statusDiv.offset();
    window.scrollTo(position.left, position.top)
  }

  function hideStatus() {
    $('#status-message').hide();
  }
});
