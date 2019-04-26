# DigitRecognition
This project is a part of our fourth year minor dissertation module. 
Our group project includes Kimberly Burke and Niall Devery.
The title of our dissertation is A neural network to process and recognise handwriting using the NIST dataset
For our project, we developed a convolutional neural network model which we trained on the NIST 
dataset. The model can in take in images of a numerical and alphabetical value to be processed 
including numbers: 0-9 and letters: A-E (all uppercase). The model
then makes a prediction on the image of what it presumes it to be and outputs the result. The
model is integrated into a python flask API for the users convenience.

To run this project simply clone it by running the command 
```
git clone: https://github.com/NiallD565/DigitRecognition
```
Then navigate to where the project was cloned and go to the model project folder then open command prompt and run the following command
```
python app.py
```
Then open browser and go to the address http://127.0.0.1:5000

Then click the button on the web page and select an image of a character you would like to predict
