import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.tree import DecisionTreeClassifier
def cleanup(var):
  return "True" if var == "1" else "False"

with open(r'dict_for_pickling.pkl', 'rb') as f:
    dict_for_unpickling = pickle.load(f)

st.write("# Segment 2 Task 1")
st.write("# Financial Anomaly Detection")

step = st.slider('step',min_value=1,max_value=743)  # 👈 this is a widget
type1 = st.selectbox("What payment are you doing?",
        (["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]),
)
amount = st.number_input("Insert the amount",min_value=0.00)
isFlaggedFraud = st.selectbox("Is it possibly fraud?",
        (["True","False"]),
)

if st.button('Submit'):
  le = dict_for_unpickling['le']
  sc = dict_for_unpickling['sc']
  dtc = dict_for_unpickling['dtc']
  rfc = dict_for_unpickling['rfc']
  lr_model = dict_for_unpickling['lr_model']
  isFlaggedFraud = 1 if isFlaggedFraud == "True" else 0
  type1 = le.transform([type1])[0]
  output = sc.transform([[step,type1,amount,isFlaggedFraud]])
  dtc_output = dtc.predict(output)[0]
  rfc_output = rfc.predict(output)[0]
  lr_output = lr_model.predict(output)[0]
  df = pd.DataFrame(
    [
        {"Classifier": "Logistic Regression", "output": bool(int(lr_output))},
        {"Classifier": "Decision Tree", "output": bool(int(dtc_output))},
        {"Classifier": "Random Forest",  "output": bool(int(rfc_output))},
    ]
  )
  st.write(df)









