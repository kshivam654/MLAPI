#import modules
from fastapi import FastAPI
from pydantic import BaseModel 
import pickle5 as pickle
import pandas as pd

app = FastAPI()

class Employee(BaseModel):
    YearsAtCompany: float #/ 1, // Float value 
    EmployeeSatisfaction: float #0.01, // Float value 
    Position:str # "Non-Manager", # Manager or Non-Manager
    Salary: int #4.0 // Ordinal 1,2,3,4,5

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/")
async def endpoint(employee: Employee):
    df = pd.DataFrame([employee.dict().values()], columns=employee.dict().keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}