{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e95caa40-2f43-4a7e-8857-15fcbcce703c",
   "metadata": {},
   "source": [
    "**DEPLOYMENT WITH STREAMLIT**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a01c62f9-222e-418c-9abd-5a12b221c80b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "import joblib\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "780f05bb-ed94-4606-9b40-0067d435cae3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\anass\\Downloads\\diabetes (2).csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "868b0aba-2136-470e-a66b-806015f9d16e",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']\n",
    "df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "83dc7124-24a0-4f5f-b8e3-df2c852a99ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns=['Outcome'])\n",
    "y = df['Outcome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "77f006e2-40f8-475e-a3ad-cb3ef77e2e91",
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_features = X.columns.tolist()\n",
    "numeric_transformer = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bbb84245-b8e0-4f58-b03e-9f93fc686cc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features)])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6da4a768-0cfb-4a92-82cb-da4498f95f7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = Pipeline([('preprocessor', preprocessor), ('clf', LogisticRegression(solver='liblinear', random_state=42))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "67f2ea7d-dc72-4542-997f-5176a88a87c4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved model\n"
     ]
    }
   ],
   "source": [
    "pipeline.fit(X, y)\n",
    "joblib.dump(pipeline, \"logreg_pipeline.pkl\")\n",
    "print(\"Saved model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c3273317-18cf-4fd5-9a53-0d035e68669d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "94c5e0f6-26c0-416d-a377-bf41ea94a1dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-11-05 08:43:52.902 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.903 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.906 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.910 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.910 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.911 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.912 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.912 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.913 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.913 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.914 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.915 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.915 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.916 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.916 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.917 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.918 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.919 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.920 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.921 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.921 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.922 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.922 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.923 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.923 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.925 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.926 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.927 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.928 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.929 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.929 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.930 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.930 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.931 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.931 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.932 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.933 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.934 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.934 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.936 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.938 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.939 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.939 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.940 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.941 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.941 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:52.942 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.title(\"Diabetes Risk Predictor\")\n",
    "\n",
    "model = joblib.load(\"logreg_pipeline.pkl\")\n",
    "\n",
    "\n",
    "preg = st.number_input(\"Pregnancies\", min_value=0, max_value=20, value=1)\n",
    "glucose = st.number_input(\"Glucose\", min_value=0.0, value=120.0)\n",
    "bp = st.number_input(\"BloodPressure\", min_value=0.0, value=70.0)\n",
    "skin = st.number_input(\"SkinThickness\", min_value=0.0, value=20.0)\n",
    "insulin = st.number_input(\"Insulin\", min_value=0.0, value=79.0)\n",
    "bmi = st.number_input(\"BMI\", min_value=0.0, value=25.0)\n",
    "dpf = st.number_input(\"DiabetesPedigreeFunction\", min_value=0.0, value=0.47)\n",
    "age = st.number_input(\"Age\", min_value=0, max_value=120, value=33)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "1fec5508-c568-4b11-9be2-a86ed403abed",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_df = pd.DataFrame([{\n",
    "    'Pregnancies': preg,\n",
    "    'Glucose': glucose,\n",
    "    'BloodPressure': bp,\n",
    "    'SkinThickness': skin,\n",
    "    'Insulin': insulin,\n",
    "    'BMI': bmi,\n",
    "    'DiabetesPedigreeFunction': dpf,\n",
    "    'Age': age\n",
    "}])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "46d91fe2-cc55-4cab-b27b-3f3c2f382f34",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-11-05 08:43:54.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:54.795 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:54.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:54.797 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-11-05 08:43:54.798 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button(\"Predict\"):\n",
    "    proba = model.predict_proba(input_df)[0,1]\n",
    "    pred = model.predict(input_df)[0]\n",
    "    st.write(f\"Predicted probability of diabetes: **{proba:.3f}**\")\n",
    "    st.write(\"Prediction:\", \"**Diabetic**\" if pred==1 else \"**Non-diabetic**\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34c2f6d0-9acc-4dd6-b543-061343571674",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
