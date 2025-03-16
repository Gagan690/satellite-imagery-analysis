# models.py
# Database models for the Satellite Imagery Analysis application

from app import db
from datetime import datetime
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """User model for authentication and tracking analyses"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Analysis(db.Model):
    """Model to store metadata about satellite image analyses"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    # Image metadata
    image_path = db.Column(db.String(256), nullable=False)
    original_filename = db.Column(db.String(256))
    image_format = db.Column(db.String(16))
    
    # Geospatial metadata
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    resolution = db.Column(db.Float)
    
    # Analysis settings
    processing_method = db.Column(db.String(64))
    features_extracted = db.Column(db.String(256))
    
    # Analysis results
    results_path = db.Column(db.String(256))
    report_path = db.Column(db.String(256))
    
    def __repr__(self):
        return f'<Analysis {self.title} - {self.timestamp}>'

class SatelliteImage(db.Model):
    """Model to store information about available satellite images"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    path = db.Column(db.String(256), nullable=False)
    description = db.Column(db.Text)
    source = db.Column(db.String(128))
    acquisition_date = db.Column(db.DateTime)
    format = db.Column(db.String(16))
    bands = db.Column(db.Integer)
    resolution = db.Column(db.Float)
    
    # Geospatial information
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    
    def __repr__(self):
        return f'<SatelliteImage {self.name}>'
