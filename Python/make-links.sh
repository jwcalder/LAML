#!/bin/bash

echo \# Python Notebooks for Linear Algebra and Machine Learning
echo 

echo \#\# Intro
for pyfile in Intro/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Optimization
for pyfile in Optimization/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

