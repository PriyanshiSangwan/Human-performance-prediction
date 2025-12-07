You can try the deployed Human Performance Prediction System here:
https://human-performance-prediction-priyanshi.onrender.com

## Project Overview:
- This project predicts the employee performance score (1 to 5) based on multiple HR-related factors. The goal is to help companies identify productivity trends, optimize workforce planning, and support data-driven HR decisions.
#### Users enter employee data such as:
- Age
- Years at Company
- Monthly Salary
- Working Hours per Week
- Projects Handled
- Overtime Hours
- Sick Days
- Remote Work Frequency
- Team Size
- Training Hours
- Promotions
- Employee Satisfaction Score
- Gender
- Education Level
- Job Title
- Resigned or Not
- Employee Segment (Cluster – generated using machine learning)
- The trained model then predicts the Performance Score of an employee from 1 (Low) to 5 (Excellent).
To analyze employee productivity and build a machine learning model that can predict performance using real corporate HR attributes.

## Machine Learning Model:
Multiple models were trained:

#### Model                          	
                                        R² Score   	RMSE	    MAE
     Random Forest Regressor   	    0.9999	  Very Low   Very Low
    Gradient Boosting Regressor	    0.989	      Low	        Low
    Linear Regression              	0.977  	Moderate	   Moderate

Random Forest Regressor was selected as the final model due to highest accuracy and lowest error.

## Feature Engineering:
- Categorical Encoding (Label & One-Hot Encoding)
- Feature Scaling
- Clustering using K-Means to add Employee Segment
- Outlier analysis
- Correlation Study

## Exploratory Data Analysis Insights: 
- Salary is the most important driver of performance
- More training hours improve productivity up to a point
- Overtime does not strongly correlate with better performance
- Mid-level employees and Managers show higher productivity
- Slightly better scores observed among Female employees
- Sales & Operations teams perform best on average

## Folder Structure:
 Employee Performance Prediction
 ┣  templates
 ┃ ┗ index.html
 ┣  static (optional for CSS)
 ┣ app.py
 ┣ model.pkl
 ┣ kmeans_model.pkl (if used)
 ┣ requirements.txt
 ┣ README.md

## How to run locally: 
- git clone https://github.com/PriyanshiSangwan/Human-performance-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
- Server runs on:
http://127.0.0.1:5000/


## Future Improvements:
- Add Authentication for HR dashboard
- Improve clustering with more employee behavior features
- Add explainability using SHAP to show feature impact
- Enhanced UI using Bootstrap/React

## Author
Priyanshi Sangwan
B.Tech – Computer Science & Engineering
Data Science Enthusiast | ML Developer
