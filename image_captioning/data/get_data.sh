#! /bin/bash
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -O ./Flickr8k_Dataset.zip
unzip ./Flickr8k_Dataset.zip -d ./Flickr8k_Dataset

wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -O ./Flickr8k_text.zip
unzip ./Flickr8k_text.zip -d ./Flickr8k_text

wget http://nlp.stanford.edu/data/glove.6B.zip -O ./glove.6B.zip
unzip ./glove.6B.zip -d ./glove.6B
