# disaster_response_pipeline
This is a project about machine learning pipeline project from the Data Scientist nana-degree program

(Link)[https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/ea367f74-3d5a-42b1-92a3-d3d3734fd369/concepts/8c0a0cd1-a2ef-4682-9a8e-c13ef7fc5e65] to Udacity Nano-degree program

# Summary
This project aims to help people consume disaster information in an efficient manner. It use natural language processing techniques classify text message into different categories based on which type of disaster it belongs to.



# How to use it?

You can open your terminal and 

#### To create a processed sqlite db
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
#### To train and save a pkl model
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
#### To deploy the application locally
python run.py


# Acknowledgement
The training dataset for building this model is coming from Figure8 and Udacity Data Scietist Nano degree program.
Kudos to Figure8 and Udacity!

