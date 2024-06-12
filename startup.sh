#!/bin/bash

# Update and install dependencies
apt-get update && apt-get install -y libx11-6

# Install any additional dependencies your application needs
# For example, you might need to install OpenCV dependencies
# apt-get install -y libopencv-dev

# Start the Flask application
# python app.py
gunicorn --bind=0.0.0.0 --timeout 600 app:app
