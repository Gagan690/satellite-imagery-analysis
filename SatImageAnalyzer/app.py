# app.py
# Main application file for Satellite Imagery Analysis Tool

import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "satellite_imagery_default_key")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///satellite_imagery.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum file upload size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize app with database extension
db.init_app(app)

# Import routes to register them with the app
with app.app_context():
    # Import models to create tables
    import models
    
    # Create database tables
    db.create_all()
    
    # Import and register routes
    from routes.main_routes import main_bp
    from routes.analysis_routes import analysis_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    
    logger.info("Satellite Imagery Analysis Tool initialized successfully")
