#!/bin/bash

echo \# Python Notebooks for Linear Algebra and Machine Learning
echo 

echo \#\# Introductory Notebooks
for pyfile in Intro/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Optimization "(Chapters 6 and 11)"
for pyfile in Optimization/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Introduction to Machine Learning "(Chapter 7)"
for pyfile in IntroML/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Principal Component Analysis "(Chapter 8)"
for pyfile in PrincipalComponentAnalysis/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Graph-Based Learning "(Chapter 9)"
for pyfile in GraphBasedLearning/*.py; do 
    ./notebook-link.sh $pyfile
done
echo 

echo \#\# Neural Networks "(Chapter 10)"
for pyfile in NeuralNetworks/*.py; do 
    ./notebook-link.sh $pyfile
done
