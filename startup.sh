#!/bin/bash

# Update the package list and install libx11-6
apt-get update
apt-get install -y libx11-6

# Start the Flask application
python -u app.py
