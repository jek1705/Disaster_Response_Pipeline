# Data Science Project: Disaster_Response_Pipeline

## Introduction

Welcome to our Data Science project on Disaster Response Pipeline, where we build a pipeline using NLP and machine learning to help improve response during disaster.

## Installation

To get started, you can clone the repo by running the following command:
git clone https://github.com/jek1705/Disaster_Response_Pipeline.git

## Project Motivation

This project is a part of my Udacity Data Science Nanodegree program, where we aim to provide help during disaster to select only the most relevant messages among the too many received at a time where it capacity to answer is the most limited.

## File Descriptions
- app	
  - templates
    - go.html  # classification result page of web app
    - master.html  # main page of web app
  - run.py  # Flask file that runs app
- data
  - disaster_categories.csv # data to process
  - disaster_messages.csv # data to process
  - process_data.py  # program to clean data and load into database
  - DisasterResponse.db  # database to save clean data to
- models
  - train_classifier.py  # program to build and train machine learning
  - classifier.pkl  # pickle file with the model trained to forecast
- README.md

## How to Interact with the Project

- This project is for educational purposes only and no contributions are expected. However, feel free to explore the data and insights provided.
1. To create a processed sqlite db:  
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
2. To train and save a pkl model:  
python train_classifier.py ../data/DisasterResponse.db classifier.pkl  
3. To deploy the application locally:
python run.py  

## Licensing, Authors, Acknowledgements, etc.

- The data used in this project is provided by Udacity (https://www.udacity.com/) and is copyright protected.
- This project is created by Jek1705 as part of the Udacity Data Science Nanodegree program.

Thank you for visiting our project and we hope you find our solution useful!
