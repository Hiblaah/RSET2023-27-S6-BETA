#!/bin/bash

echo "Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "Error: npm install failed. Please ensure Node.js is installed."
    exit 1
fi

echo "Dependencies installed successfully!"
