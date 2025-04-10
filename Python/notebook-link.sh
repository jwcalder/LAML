#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: notebook-link.sh <py file>"
    echo "Creates and prints out link to ipynb notebook."
    exit 1
fi
pyfile=$1

url=https://colab.research.google.com/github/jwcalder/LAML/blob/main/Python/"$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"

echo "* [$(basename "$pyfile" .py)]("$url")"
