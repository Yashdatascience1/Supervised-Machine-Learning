import matplotlib
import streamlit as st
import streamlit.components.v1 as components
import pickle 
import pandas as pd
import numpy as np
import shap
from catboost import Pool
raw_text = u"\u20B9"

def cat_col(df):
        cat_col = df.select_dtypes(exclude=['int','float']).columns.tolist()
        return cat_col

attrib_info = """
#### 
- Total_Experience : Total industry experience
- Relevant_experience : Total experience in the field applied for (past work experience that is relevant to the job)
- Department : Department name of current company
- Role : Role in the current company
- Industry : Industry name of current field
- Organization : Organization name
- Designation : Designation in current company
- Education : Education
- Graduation_Specialization : Specialization subject in graduation
- PG_Specialization : Specialization subject in Post-Graduation
- PHD_Specialization : Specialization subject in Post-Graduation
- Current_CTC : Current CTC
- Inhand_Offer : Holding any offer in hand (Y: Yes, N:No)
- Last_Appraisal_Rating : Last Appraisal Rating in current company
- No_Of_Companies_worked : No. of companies worked till date
- Number_of_Publications : Number of papers published
- Certifications : Number of relevant certifications completed
- International_degree_any : Hold any international degree (1: Yes, 0: No)
- Expected_CTC : Expected CTC (Final CTC offered by Delta Ltd.)
"""

@st.cache
def load_model(model_file):
    model = pickle.load(open(model_file,'rb'))
    return model

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def run_Cap_ML():
    st.subheader("ML section")
    data = pd.read_csv('./Data/Data_for_deployment.csv')
    cat_features = ['Department',
 'Role',
 'Industry',
 'Organization',
 'Designation',
 'Graduation_Specialization',
 'PG_Specialization',
 'PHD_Specialization',
 'Last_Appraisal_Rating']
    dept_unique = tuple(data['Department'].unique())
    role_unique = tuple(data['Role'].unique())
    industry_unique = tuple(data['Industry'].unique())
    org_unique = tuple(data['Organization'].unique())
    designation_unique = tuple(data['Designation'].unique())
    inhand_offer_unique = tuple(data['Inhand_Offer'].unique())
    last_appraisal_rating_unique = tuple(data['Last_Appraisal_Rating'].unique())
    education_unique = tuple(data['Education'].unique())
    Graduation_Specialization_unique = tuple(data['Graduation_Specialization'].unique())
    PG_Specialization_unique = tuple(data['PG_Specialization'].unique())
    PHD_Specialization_unique = tuple(data['PHD_Specialization'].unique())
    model = load_model('CatBoost_Regressor.pkl')
    with st.expander("Attribute information"):
        st.markdown(attrib_info)

    with st.expander("Enter your professional details"):
        st.subheader("Professional details")
        Total_Experience = st.slider("What is your total experience(in years)?",min_value=0,max_value=50,value=5,step=1)
        Relevant_experience = st.slider("What is your experience in the field you are applying for(in years)?",min_value=0,max_value=50,value=5,step=1)
        Department = st.selectbox("What is your current department of work?",dept_unique)
        Role = st.selectbox("What is your current role at work?",role_unique)
        Industry = st.selectbox("Industry you work for",industry_unique)
        Organization = st.selectbox('Organization you work for',org_unique)
        Designation = st.selectbox('Your designation',designation_unique)
        Current_CTC = st.number_input("Enter your current annual salary",min_value=0,max_value=10000000,value=0,step=10000)
        Inhand_Offer = st.radio("Do you have an offer in hand",inhand_offer_unique)
        Last_Appraisal_Rating = st.selectbox("What was your last appraisal rating?",last_appraisal_rating_unique)
        No_Of_Companies_worked = st.slider("How many companies have you worked for till now?",min_value=0,max_value=6,value=0,step=1)
    
    with st.expander("Enter your educational details"):
        st.subheader("Educational details")
        Education = st.selectbox("What is your educational level?",education_unique)
        if Education == 'Under Grad':
            Education_var = 0
        elif Education == 'Grad':
            Education_var = 1
        elif Education == 'PG':
            Education_var = 2
        else:
            Education_var = 3
        Graduation_Specialization = st.selectbox("What was your specialization subject in graduation?",Graduation_Specialization_unique)
        PG_Specialization = st.selectbox("What was your specialization subject in post-graduation?",PG_Specialization_unique)
        PHD_Specialization = st.selectbox("What was your specialization subject in PHD?",PHD_Specialization_unique)
        Certifications = st.slider("How many certification courses have you done?",min_value=0,max_value=5,step=1)
        International_degree = st.radio("Do you have any international degree?",("Yes","No"))
        if International_degree == "Yes":
            International_degree_any = 1
        else:
            International_degree_any = 0
        
    with st.expander("Your selection"):
        result = {"Total experience":Total_Experience,"Relevant experience":Relevant_experience,"Department":Department,
                "Role":Role,"Industry":Industry,"Organization":Organization,"Designation":Designation,
                "Education":Education_var,"Graduation Specialization subject":Graduation_Specialization,
                "PG Specialization subject":PG_Specialization,"PHD Specialization subject":PHD_Specialization,
                "Current salary":Current_CTC,"Inhand offer":Inhand_Offer,"Last appraisal rating":Last_Appraisal_Rating,
                "# of companies worked":No_Of_Companies_worked,
                "# of certifications":Certifications,"International degree":International_degree_any}
        st.json(result)
    
    encoded_result = list(result.values())

    with st.expander("Prediction Results"):
        single_sample = pd.DataFrame(data=[encoded_result],columns=['Total_Experience','Relevant_experience','Department',
        'Role','Industry','Organization','Designation','Education','Graduation_Specialization','PG_Specialization', 'PHD_Specialization',
        'Current_CTC', 'Inhand_Offer', 'Last_Appraisal_Rating','No_Of_Companies_worked', 'Certifications', 'International_degree_any'])        
        prediction = model.predict(single_sample)
        df1 = pd.DataFrame(data=[prediction],columns=["Expected_CTC"])
        st.info("Your expected CTC will be {}{:,.0f}".format(raw_text,np.round(prediction[0]),2))

        shap_values = model.get_feature_importance(Pool(single_sample,label=df1,cat_features=cat_features),type='ShapValues')

        expected_value = shap_values[0,-1]
        shap_values = shap_values[:,:-1]
        st_shap(shap.force_plot(expected_value,shap_values[0,:],single_sample))



