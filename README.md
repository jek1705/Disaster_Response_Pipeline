# Data Science Project: Disaster_Response_Pipeline

## Introduction

Welcome to our Data Science project on Disaster Response Pipeline, where we build a pipeline using NLP and machine learning to help improve response during disaster.

## Installation

To get started, you can clone the repo by running the following command:
git clone https://github.com/jek1705/Disaster_Response_Pipeline.git

## Project Motivation

This project is a part of my Udacity Data Science Nanodegree program, where we aim to provide help during disaster to select only the most relevant messages among the too many received at a time where it capacity to answer is the most limited.

## File Descriptions

•	app	
o	templates
	go.html  # classification result page of web app
	master.html  # main page of web app
o	run.py  # Flask file that runs app
•	data
o	disaster_categories.csv # data to process
o	disaster_messages.csv # data to process
o	process_data.py  # program to clean data and load into database
o	DisasterResponse.db  # database to save clean data to
•	models
o	train_classifier.py  # program to build and train machine learning
o	classifier.pkl  # pickle file with the model trained to forecast
•	README.md




## How to Interact with the Project

This project is for educational purposes only and no contributions are expected. However, feel free to explore the data and insights provided.
To run the APP, from command python run.py then go to http://127.0.0.1:3000/

## Licensing, Authors, Acknowledgements, etc.

- The data used in this project is provided by Udacity (https://www.udacity.com/) and is copyright protected.
- This project is created by Jek1705 as part of the Udacity Data Science Nanodegree program.

Thank you for visiting our project and we hope you find our solution useful!
