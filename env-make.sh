#!/usr/bin/env bash

echo >.env
echo >>.env PROJECT_ROOT=`pwd`
echo >>.env TESSDATA_PREFIX=/Users/thst/homebrew/Cellar/tesseract-lang/4.1.0/share/tessdata
read -r -p "Enter your GOOGLE_API_KEY: " GOOGLE_API_KEY
echo >>.env GOOGLE_API_KEY=$GOOGLE_API_KEY
