import streamlit as st
import streamlit.components.v1 as stc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import pickle 
import shap
from catboost import Pool
raw_text = u"\u20B9"

html_temp = """
		<div style="background-color:#FA8072;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Determining the CTC to be offered to a new candidate </h1>
		</div>"""
def main():
    stc.html(html_temp)
    menu = ["Home","EDA","ML"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.markdown("""
                    ### Title: CTC prediction app
                    #### Dataset source
                    This dataset contains educational and professional information about the candidate applying

                        - https://drive.google.com/file/d/1N_9Bg6o_3A_bmvsx6fh6ohJnrmWUNOI4/view
 
                    #### App contents
                    - EDA -  Exploratory analysis of Data
                    - ML - Predict the CTC to be offered to a new candidate

                        """)
    elif choice == "EDA":
        run_Cap_EDA()
    elif choice == "ML":
        run_Cap_ML()

def run_Cap_EDA():
    def cat_col(df):
        cat_col = df.select_dtypes(exclude=['int','float']).columns.tolist()
        return cat_col

    def cont_col(df):
        cont_col = df.select_dtypes(include=['int','float']).columns.tolist()
        return cont_col
    st.subheader("Exploratory data analysis")
    df = pd.read_csv('./Data/expected_ctc.csv')
    df.rename(columns={"Total_Experience_in_field_applied":"Relevant_experience","Curent_location":"Current_location"},inplace=True)
    submenu = st.sidebar.selectbox("Submenu",('Descriptive','Plots'))
    if submenu == "Descriptive":
        st.dataframe(df.head())

        with st.expander("Shape of the data"):
            st.dataframe(pd.DataFrame(data=[df.shape[0],df.shape[1]],columns=['Count'],index=['No. of rows','No. of columns']).astype(str))
        
        with st.expander("Distribution of categorical and continous variables"):
            st.dataframe(pd.DataFrame(data=[len(cat_col(df)),len(cont_col(df))],columns=["No. of features"],index=["Categorical features","Numerical features"]).astype(str))

        with st.expander("Data types"):
            st.dataframe(pd.DataFrame(df.dtypes,columns=["Data types"]).astype(str))
        
        with st.expander("Descriptive summary"):
            st.dataframe(pd.DataFrame(df.describe()).astype(str))
        
        with st.expander("Missing data"):
            missing_data=df.isnull().sum().reset_index()
            missing_data.rename(columns={"index":"Feature name",0:"Number of missing values"},inplace=True)
            missing_data.set_index('Feature name')
            missing_data=missing_data.sort_values(by="Number of missing values",ascending=False)
            st.dataframe(missing_data.astype(str))
        
        with st.expander("Outliers"):
            def number_of_outliers(df):
                IQR = df.quantile(0.75) - df.quantile(0.25)
                Number = ((df > (df.quantile(0.75)+(1.5*IQR))) | (df < (df.quantile(0.25)-(1.5*IQR)))).sum()
                Outlier = ((df > (df.quantile(0.75)+(1.5*IQR))) | (df < (df.quantile(0.25)-(1.5*IQR))))
                dataframe = pd.DataFrame({"Number of Outliers":Number})
                return (dataframe.sort_values(by="Number of Outliers",ascending=False))
            
            outliers = number_of_outliers(df)
            st.dataframe(outliers.astype(str))
        
        with st.expander("Skewness"):
            cont_col = cont_col(df)
            skewness = pd.DataFrame(df[cont_col].skew(),columns=["Skewness"])

            Symmetric=skewness[((skewness["Skewness"]<=0.5) & (skewness["Skewness"]>=-0.5))].index
            Moderately_symmetric = skewness[(((skewness["Skewness"]<=-0.5) & (skewness["Skewness"]>=-1))| ((skewness["Skewness"]<=1) & (skewness["Skewness"]>=0.5)))].index
            Asymmetric = skewness[((skewness["Skewness"]<=-1) | (skewness["Skewness"]>=1))].index

            col1,col2,col3 = st.columns(3)

            with col1:
                st.subheader("Symmetric features")
                st.dataframe(Symmetric.astype(str))
            
            with col2:
                st.subheader("Asymmetric features")
                st.dataframe(Asymmetric.astype(str))
            
            with col3:
                st.subheader("Moderately symmetric features")
                st.dataframe(Moderately_symmetric.astype(str))
                



        
    elif submenu == 'Plots':
        st.subheader("Plots")
        df1 = df[["Current_CTC","Expected_CTC"]]
        df1 = pd.melt(df1)
        with st.expander("Analysis of Expected CTC Vs. Current CTC"):
            fig = px.histogram(df1,x="value",color="variable",marginal='box')
            st.plotly_chart(fig)
        
        with st.expander("Average current CTC by department"):
            fig=plt.figure(figsize=(14,4))
            p=sns.barplot(x=df["Department"],y=df["Current_CTC"],estimator=np.mean)
            p.set_title("Average current CTC by department",fontdict={'weight':'bold'})
            st.pyplot(fig)
        
        with st.expander("Analysis by range of experience"):
            data = df.copy()
            data["Total_Experience_range"]=pd.cut(data["Total_Experience"],bins=[0,2,6,11,16,21,25],
                              labels=['0-1','2-5','6-10','11-15','16-20','21-25'])
            r1=range(data["Total_Experience_range"].nunique())
            v1 = sorted(data["Total_Experience_range"].value_counts(1).values*100)
            v2 = sorted(data["Total_Experience_range"].value_counts())
            fig=plt.figure()
            sns.countplot(x=data["Total_Experience_range"],order=data["Total_Experience_range"].value_counts().sort_values().index)
            for (i,j,k) in zip(r1,v1,v2):
                plt.text(x=i-0.3,y=k+46,s="{:.1f}%".format(j),color="black",fontsize=10,fontweight="bold")
            plt.ylabel('No. of candidates')
            plt.xlabel('Range of Total experience')
            plt.title("Percentage of candidates according to the total experience")
            st.pyplot(fig)

            data["Relevant_experience_range"]=pd.cut(data["Relevant_experience"],bins=[0,2,6,11,16,21,25],
                              labels=['0-1','2-5','6-10','11-15','16-20','21-25'])
            r2=range(data["Relevant_experience_range"].nunique())
            v3 = sorted(data["Relevant_experience_range"].value_counts(1).values*100)
            v4 = sorted(data["Relevant_experience_range"].value_counts())
            fig1 = plt.figure()
            sns.countplot(x=data["Relevant_experience_range"],order=data["Relevant_experience_range"].value_counts().sort_values().index)
            for (i,j,k) in zip(r2,v3,v4):
                plt.text(x=i-0.3,y=k+46,s="{:.1f}%".format(j),color="black",fontsize=10,fontweight="bold")
            plt.ylabel('No. of applicants')
            plt.xlabel('Relevant experience')
            plt.title("Percentage of candidates according to the total experience in relevant field")
            plt.show()
            st.pyplot(fig1)
        
        with st.expander("Analysis of expected CTC by range of experience"):
            d1=data.groupby("Total_Experience_range").agg({"Expected_CTC":"mean"})["Expected_CTC"]
            fig2=plt.figure(figsize=(8,6))
            sns.barplot(x="Total_Experience_range",y="Expected_CTC",data=data,ci=False)
            for (i,j) in enumerate(d1):
                plt.text(x=i-0.4,y=j+40000,s="{}{:,.0f}".format(raw_text,j),color="black",fontsize=10,fontweight="bold")
            plt.title("Average expected CTC by total experience")
            plt.show()
            st.pyplot(fig2)

            d4=data.groupby("Relevant_experience_range").agg({"Current_CTC":"mean"})["Current_CTC"]
            fig3=plt.figure(figsize=(8,6))
            sns.barplot(x="Relevant_experience_range",y="Current_CTC",data=data,ci=False)
            for (i,j) in enumerate(d4):
                plt.text(x=i-0.4,y=j+40000,s="{}{:,.0f}".format(raw_text,j),color="black",fontsize=10,fontweight="bold")
            plt.title("Average current CTC by total years of relevant experience")
            plt.show()
            st.pyplot(fig3)

        with st.expander("Analysis of numerical columns"):
            cont_col1 = tuple(cont_col(df))
            choice = st.selectbox("Choose a numerical feature",cont_col1)
            col1,col2 = st.columns(2)

            with col1:
                trace = go.Histogram(x=df[choice],nbinsx=50)
                data = [trace]
                layout = go.Layout(title="Histogram of {}".format(choice))
                fig = go.Figure(data=data,layout=layout)
                st.plotly_chart(fig)
        
            with col2:
                trace = go.Box(y=df[choice],name=choice)
                data = [trace]
                layout = go.Layout(title="Boxplot of {}".format(choice),yaxis={"title":choice})
                fig = go.Figure(data=data,layout=layout)
                st.plotly_chart(fig,use_container_width=True)
        
        
        with st.expander("Proportion of categories in categorical features"):
            cat_col1 = cat_col(df)
            cat_col1 = tuple(cat_col1)
            choice2 = st.selectbox("Select one category to count the proportion of levels for",cat_col1)

            val_counts=pd.DataFrame(df[choice2].value_counts().values,columns=["Count"],index=df[choice2].value_counts().index).reset_index()
            val_counts.rename(columns={"index":choice2},inplace=True)           
            fig=px.bar(data_frame=val_counts,x=choice2,y="Count",barmode='group')
            st.plotly_chart(fig)
        
        with st.expander("Distribution of current CTC across different categories"):
            cat_col2 = cat_col(df)
            cat_col2 = tuple(cat_col2)
            choice3 = st.selectbox("Select one category",cat_col2)
            fig=sns.catplot(x=choice3,y='Current_CTC',data=df,kind='box')
            plt.xticks(rotation=90)
            st.pyplot(fig)
        
        with st.expander("Distribution of expected CTC across different categories"):
            cat_col3 = cat_col(df)
            cat_col3 = tuple(cat_col3)
            choice4 = st.selectbox("Select one category from the options",cat_col3)
            fig=sns.catplot(x=choice4,y='Expected_CTC',data=df,kind='box')
            plt.xticks(rotation=90)
            st.pyplot(fig)

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
    stc.html(shap_html, height=height)


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
    
if __name__ == '__main__':
    main()