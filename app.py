import streamlit as st
import numpy as np
import cv2
from joblib import load

st.title("Yelp Review Sentiment Classifer")
st.header("This app detects the sentiment of your Yelp review! Enter your text to try it out!")
model = load("model.joblib")

user_review = st.text_input("Enter your review here.")
#sentiment = model.predict(user_review)
#st.write("The predicted sentiment is:" + sentiment)









#EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

#f = st.file_uploader("Upload Image")
#if f is not None: 
  # Extract and display the image
#  file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
#  image = cv2.imdecode(file_bytes, 1)
#  st.image(image, channels="BGR")
  # Run the model
#  scores = model.predict(image)
  # Print results and plot score
#  st.write(f"The predicted emotion is: {EMOTIONS[scores.argmax()]}")




#model = load("model.joblib")
#st.write ("Model uploaded!")

#def get_user_input():
#  blood_pressure = st.number_input("What is your blood pressure?")
#  heart_rate = st.number_input("What is your maximum heart rate during exercise in beats per minute?")
#  input_features = [[blood_pressure, heart_rate]]
#  return input_features

#def make_prediction(model, input):
#  return model.predict(input)

#def get_app_response(prediction):
#  if prediction == 1:
#    st.write("The model predicts you have heart disease.")
#  elif prediction == 0:
#    st.write("The model predicts you do not have heart disease.")
#  else:
#    st.write ("No results yet")

#input_features = get_user_input()
#prediction = make_prediction(model, input_features)
#get_app_response(prediction)



#st.button('Press Start')
#st.selectbox('Pick your user', ['Student', 'Parent', 'Teacher'])
#st.slider('Pick your age', 0, 100)

#import pandas as pd
#import numpy as np

#st.write("My line chart")
#df = pd.DataFrame({
#     'column 1': [0, 1, 2, 3],
#     'column 2': [100, 50, 200, 100],
# })
#st.line_chart(df)


#rand = np.random.normal(1, 2)
#fig, ax = plt.subplots()
#ax.hist(rand, bins=15)
#st.pyplot(fig)








#st.write("Welcome!")
#st.image("logo.jpg")







#import pandas as pd




#st.title("This is my title.")
#st.header("This is my header.")
#st.subheader("This is my subheader.")
#st.caption("This is my caption.")

#st.write(pd.DataFrame({
#     'column 1': [1, 2, 3, 4],
#     'column 2': [100, 200, 300, 400],
# }))




