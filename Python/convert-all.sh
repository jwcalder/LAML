#!/bin/bash

for pyfile in */*.py; do 
    ./to-ipynb.sh $pyfile
done

cp QRCodes/* ~/Overleaf/LAML/QRCodes/
