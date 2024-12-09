import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("mental_health_diagnosis_treatment_.csv")


# binomial gender
df_b = df.copy()
df_b['Gender'] = df_b['Gender'].map({'Female':1, 'Male':0})


# make rating scaled features binary
features = ['Symptom Severity (1-10)', 'Sleep Quality (1-10)', 'Mood Score (1-10)', 'Stress Level (1-10)', 'Treatment Progress (1-10)']

for feat in features:
    df_b[feat] = (df_b[feat] > 5).astype(int)
    
df_b["Adherence to Treatment (%)"] = (df_b["Adherence to Treatment (%)"] > 80).astype(int)



# use labelEncoder to numericalize categorical values
encoder = LabelEncoder()

encode = ["Medication", "Therapy Type", "AI-Detected Emotional State", "Diagnosis"]
encode_n = ["Med_n", "Therapy_n", "Emot_n", "Diag_n"]

for i in range(len(encode)):
    df_b[encode_n[i]] = encoder.fit_transform(df_b[encode[i]])
    

# drop start date
df_b = df_b.drop("Treatment Start Date", axis="columns")



# Drop deteriorated 
df1 = df_b.copy()
df2 = df_b.copy()

# drop if outcome=deteriorated 
df1 = df1[df1['Outcome'] != 'Deteriorated']

# drop if outcome=improve
df2 = df2[df2['Outcome'] != 'Improved']

df1['Out_b'] = df1['Outcome'].map({'No Change': 0, 'Improved':1})
df2['Out_b'] = df2['Outcome'].map({'No Change': 0, 'Deteriorated':1})

feats = features + encode_n