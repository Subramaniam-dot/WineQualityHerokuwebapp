import streamlit as st
import joblib,os
import numpy as np
import pickle
import pandas as pd

from PIL import Image
#import spacy





#Load our models
#def load_prediction_models(model_file):
   # loaded_models = joblib.load(open(os.path.join(model_file),'rb'))
   # return loaded_models

loaded_model = pickle.load(open('trained_model.sav','rb'))

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def wine_pred(input_data):

    

    prediction = loaded_model.predict(input_data)
    return prediction

    #if(prediction[0]==1):
      #return 'Good Quality Wine'
    #else:
      #return 'Bad Quality Wine'


def main():
    """Wine Prediction System"""
    st.title("Quality of Wine Predictive System")
    st.subheader("Machine Learning on Physicochemical properties of Wine")
    
    st.text('--> Below Dataset is used to train the Random Classifier Machine Learning Model and helps us predict the quality of wine.')
    st.text('--> If the qualiy of wine is said to be 7 and above, then it is a Good Quality wine. If not, it is a Bad Quality Wine')
    wine = pd.read_csv('winequality-red.csv')
    st.write(wine)
    activities = ["Prediction",]

    choice = st.sidebar.selectbox("Select",activities)

    if choice == 'Prediction':
        st.info('Prediction with ML')

        #news_text = st.text_area("Enter Text","Type Here")
        fixed_acidity = st.number_input('Enter fixed acidity',step=1e-6,format="%.5f")
        volatile_acidity =  st.number_input('Enter volatile acidity',step=1e-6,format="%.5f")
        citric_acid =  st.number_input('Enter citric acid',step=1e-6,format="%.5f")
        residual_sugar= st.number_input('Enter residual sugar',step=1e-6,format="%.5f")
        chlorides = st.number_input('Enter chlorides',step=1e-6,format="%.5f")
        free_sulfur_dioxide = st.number_input('Enter free sulfur dioxide',step=1e-6,format="%.5f")
        total_sulfur_dioxide = st.number_input('Enter total sulfur dioxide',step=1e-6,format="%.5f")
        density = st.number_input('Enter density',step=1e-6,format="%.5f")
        ph = st.number_input('Enter Ph',step=1e-6,format="%.5f")
        sulphates = st.number_input('Enter sulphates',step=1e-6,format="%.5f")
        alcohol = st.number_input('Enter Alcohol content',step=1e-6,format="%.5f")
        
        input_data=''

        input_data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,ph,sulphates,alcohol]

# changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

#reshape the data as we are the predicting the label for only one instance

        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
       


#fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
        all_ml_model = ['Random Forest Classifier']
        model_choice = st.selectbox('Choose model',all_ml_model)
        prediction_labels = {'Bad Quality Wine':0 , 'Good Quality Wine':1}
        if st.button('classify'):
            
            if model_choice == 'Random Forest Classifier':
                
                result = wine_pred(input_data_reshaped)
                #prediction = loaded_model.predict(input_data_reshaped)
                st.write(result)
                final_results = get_keys(result,prediction_labels)
                st.success(final_results)
           
    
    
    st.subheader("Data Analysis")
    image1 = Image.open('img/Al_vs_Q.png')
    image2 = Image.open('img/CA_vs_Q.png')
    image3 = Image.open('img/Ch_vs_Q.png')
    image4 = Image.open('img/D_vsQ.png')
    image5 = Image.open('img/FA_vs_Q.png')
    image6 = Image.open('img/FSD_vs_Q.png')
    image7 = Image.open('img/PH_vs_Q.png')
    image8 = Image.open('img/RS_vs_Q.png')
    image9 = Image.open('img/S_vs_Q.png')
    image10 = Image.open('img/TSD_vs_Q.png')
    image11 = Image.open('img/VA_vs_Q.png')
    image12 = Image.open('img/count-quality.png')
    image13 = Image.open('img/Corr_Heatmap.png') 
    
    st.text('Heat Map for Correlation Matrix')
    st.image(image13)


    st.text('Box Plot of Quality VS Other Physiochemical Properties')
    st.image([image1,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11])
    
    st.text('Number of Distinct Counts for Quality')
    st.image(image12)

    st.sidebar.header('About Data  set:')

    st.sidebar.text('This dataset is also available from the UCI machine learning repository, https://archive.ics.uci.edu/ml/datasets/wine+quality. Dataset is taken from Kaggle.')
    st.sidebar.header('Acknowledgements:') 
    st.sidebar.text('P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.')

    st.sidebar.header('Relevant Publications:') 
    st.sidebar.text('P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.In Decision Support Systems, Elsevier, 47(4):547-553, 2009.')
      
    st.sidebar.write('Done By [Subramaniam](https://subramaniam-dot.github.io)')





if __name__ =='__main__':
 main()
