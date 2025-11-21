1. Project Title
SMS Spam Detection System Using Machine Learning

2. Overview of the Project
This project is a Machine Learning–based system designed to classify SMS messages as Spam or Ham (Not Spam).
It uses Natural Language Processing (NLP) techniques to convert text into numerical features and a supervised learning algorithm to detect spam with high accuracy.

The purpose of the project is to automate spam detection, improve user safety, and explore text classification techniques.

3. Features

 Classifies SMS as Spam or Ham

 Displays spam probability score

 Works in command-line mode and interactive mode

 Cleans and preprocesses text automatically

 Includes EDA (Exploratory Data Analysis) for understanding dataset

 Uses TF-IDF vectorization for text processing

 Model saved using .pkl files for quick reuse

 Easy to extend into a web or mobile application

4. Technologies / Tools Used

1. Languages

Python 3.10+

2. Libraries

scikit-learn

Pandas

NumPy

Matplotlib

Joblib

3. Tools

VS Code

Git & GitHub

Environment (env)

5. Steps to Install & Run the Project
1. Clone the repository
     git clone https://github.com/nishant25bce10660-cell/SMS_Spam_Detection.git
       cd sms-spam-detection

2. Create a virtual environment
    
3. Activate the environment

4. Install dependencies
   pip install -r requirements.txt

5. Add the SMS Spam dataset
    spam.csv inside data/ folder

6. Train the model

7. Run predictions
   
Interactive mode
 python predict.py

6. Instructions for Testing
 : Test with spam messages:

"Congratulations! You won a free prize"
"Claim your lottery reward now"
"Get a loan approved instantly"

: Test with ham messages:

"Hey, are you coming today?"
"Call me when you're free"
"Where should we meet?"
