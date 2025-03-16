# routes/main_routes.py
# Main routes for the Satellite Imagery Analysis application

import os
from datetime import datetime
from flask import Blueprint, render_template, current_app, send_from_directory
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Define the blueprint
main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
def index():
    """Render the home page."""
    try:
        return render_template('index.html', now=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return render_template('error.html', error=str(e))

@main_bp.route('/documentation')
def documentation():
    """Render the documentation page."""
    try:
        return render_template('documentation.html', now=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering documentation page: {str(e)}")
        return render_template('error.html', error=str(e))

@main_bp.route('/favicon.ico')
def favicon():
    """Serve the favicon."""
    try:
        return send_from_directory(os.path.join(current_app.root_path, 'static'),
                                'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except Exception as e:
        logger.error(f"Error serving favicon: {str(e)}")
        return '', 404

@main_bp.route('/robots.txt')
def robots():
    """Serve the robots.txt file."""
    try:
        return send_from_directory(os.path.join(current_app.root_path, 'static'),
                                'robots.txt', mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error serving robots.txt: {str(e)}")
        return '', 404

@main_bp.route('/privacy')
def privacy():
    """Render the privacy policy page."""
    try:
        return render_template('privacy.html', now=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering privacy page: {str(e)}")
        return render_template('error.html', error=str(e))

@main_bp.route('/terms')
def terms():
    """Render the terms of service page."""
    try:
        return render_template('terms.html', now=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering terms page: {str(e)}")
        return render_template('error.html', error=str(e))

@main_bp.app_errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path}")
    return render_template('404.html', now=datetime.now()), 404

@main_bp.app_errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"500 error: {str(e)}")
    return render_template('500.html', now=datetime.now()), 500

# Add context processor to provide utility functions and variables to templates
@main_bp.context_processor
def utility_processor():
    """Add utility functions and variables to template context."""
    return {
        'now': datetime.now()
    }
