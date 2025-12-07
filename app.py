# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# --------------------------------------------------
# 1. Load the trained Random Forest model
# --------------------------------------------------
MODEL_PATH = "employee_performance_final_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


# --------------------------------------------------
# 2. Helper: encode one row exactly like training
# --------------------------------------------------
def preprocess_input(form_data):
    """
    Convert raw form inputs into a feature vector
    with the SAME order and encoding used in training.
    """

    # ----- Numeric / direct fields from the form -----
    # Make sure your HTML form has these 'name' attributes:
    # Age, Years_At_Company, Monthly_Salary, Work_Hours_Per_Week,
    # Projects_Handled, Overtime_Hours, Sick_Days, Remote_Work_Frequency,
    # Team_Size, Training_Hours, Promotions,
    # Employee_Satisfaction_Score, Resigned, Cluster

    age = int(form_data.get("Age"))
    years_at_company = int(form_data.get("Years_At_Company"))
    monthly_salary = float(form_data.get("Monthly_Salary"))
    work_hours = float(form_data.get("Work_Hours_Per_Week"))
    projects_handled = int(form_data.get("Projects_Handled"))
    overtime_hours = float(form_data.get("Overtime_Hours"))
    sick_days = int(form_data.get("Sick_Days"))
    remote_freq = float(form_data.get("Remote_Work_Frequency"))
    team_size = int(form_data.get("Team_Size"))
    training_hours = float(form_data.get("Training_Hours"))
    promotions = int(form_data.get("Promotions"))
    satisfaction = float(form_data.get("Employee_Satisfaction_Score"))
    resigned = int(form_data.get("Resigned"))   # 0 = No, 1 = Yes
    cluster = int(form_data.get("Cluster"))     # from your clustering step (or 0)

    # ----- Categorical fields from the form -----
    # Department, Gender, Job_Title, Education_Level
    department = form_data.get("Department")
    gender = form_data.get("Gender")
    job_title = form_data.get("Job_Title")
    education_level = form_data.get("Education_Level")

    # ----- Encode Education_Level (LabelEncoder logic) -----
    # In training you used LabelEncoder() on:
    # ['Bachelor','High School','Master','PhD'] (alphabetical order)
    edu_map = {
        "Bachelor": 0,
        "High School": 1,
        "Master": 2,
        "PhD": 3
    }
    education_encoded = edu_map.get(education_level, 0)  # default to Bachelor if unknown

    # ----- One-hot encoding for Department (drop_first=True) -----
    dept_cols = [
        "Department_Engineering", "Department_Finance", "Department_HR",
        "Department_IT", "Department_Legal", "Department_Marketing",
        "Department_Operations", "Department_Sales"
        # base category (dropped): Customer Support
    ]
    dept_vector = dict.fromkeys(dept_cols, 0)

    if department == "Engineering":
        dept_vector["Department_Engineering"] = 1
    elif department == "Finance":
        dept_vector["Department_Finance"] = 1
    elif department == "HR":
        dept_vector["Department_HR"] = 1
    elif department == "IT":
        dept_vector["Department_IT"] = 1
    elif department == "Legal":
        dept_vector["Department_Legal"] = 1
    elif department == "Marketing":
        dept_vector["Department_Marketing"] = 1
    elif department == "Operations":
        dept_vector["Department_Operations"] = 1
    elif department == "Sales":
        dept_vector["Department_Sales"] = 1
    # if department == "Customer Support", all 0 (base)

    # ----- One-hot for Gender (drop_first=True, base = Female) -----
    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0

    # ----- One-hot for Job_Title (drop_first=True, base = Analyst) -----
    job_cols = [
        "Job_Title_Consultant", "Job_Title_Developer",
        "Job_Title_Engineer", "Job_Title_Manager",
        "Job_Title_Specialist", "Job_Title_Technician"
    ]
    job_vector = dict.fromkeys(job_cols, 0)

    if job_title == "Consultant":
        job_vector["Job_Title_Consultant"] = 1
    elif job_title == "Developer":
        job_vector["Job_Title_Developer"] = 1
    elif job_title == "Engineer":
        job_vector["Job_Title_Engineer"] = 1
    elif job_title == "Manager":
        job_vector["Job_Title_Manager"] = 1
    elif job_title == "Specialist":
        job_vector["Job_Title_Specialist"] = 1
    elif job_title == "Technician":
        job_vector["Job_Title_Technician"] = 1
    # if "Analyst", all remain 0 (base category)

    # --------------------------------------------------
    # Build final feature vector in EXACT training order
    # --------------------------------------------------
    feature_list = [
        age,
        years_at_company,
        education_encoded,
        monthly_salary,
        work_hours,
        projects_handled,
        overtime_hours,
        sick_days,
        remote_freq,
        team_size,
        training_hours,
        promotions,
        satisfaction,
        resigned,
        cluster,

        dept_vector["Department_Engineering"],
        dept_vector["Department_Finance"],
        dept_vector["Department_HR"],
        dept_vector["Department_IT"],
        dept_vector["Department_Legal"],
        dept_vector["Department_Marketing"],
        dept_vector["Department_Operations"],
        dept_vector["Department_Sales"],

        gender_male,
        gender_other,

        job_vector["Job_Title_Consultant"],
        job_vector["Job_Title_Developer"],
        job_vector["Job_Title_Engineer"],
        job_vector["Job_Title_Manager"],
        job_vector["Job_Title_Specialist"],
        job_vector["Job_Title_Technician"],
    ]

    return np.array([feature_list])   # shape (1, 31)


# --------------------------------------------------
# 3. Routes
# --------------------------------------------------
@app.route("/")
def home():
    # Your templates/index.html should contain the form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Convert form values to model-ready features
    features = preprocess_input(request.form)

    # Predict performance score (regression)
    pred_score = model.predict(features)[0]

    # You can round or cast to int, depending on how you want to display
    pred_score_rounded = round(float(pred_score), 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Performance Score: {pred_score_rounded}"
    )


if __name__ == "__main__":
    app.run(debug=True)
