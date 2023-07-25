#!/bin/bash
# Check if an argument has been provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./upload-to-medium.sh <input_file>"
    exit 1
fi

# Define the input file
INPUT_FILE=$1

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "File $INPUT_FILE does not exist."
    exit 1
fi

# Generate the output file name by replacing the .md extension with .html
OUTPUT_FILE="${INPUT_FILE%.md}.html"

# Use pandoc to convert the markdown file to html
pandoc $INPUT_FILE -o $OUTPUT_FILE

# Use md-publisher to publish the html file
md-publisher publish $OUTPUT_FILE






