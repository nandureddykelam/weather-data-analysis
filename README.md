 Weather Data Analysis and Temperature Prediction using Machine Learning

 Overview

This project analyzes weather data using Python and applies machine learning techniques to predict temperature based on atmospheric conditions. It focuses on extracting meaningful insights through data analysis and building a predictive model.

 Objectives

* Analyze weather patterns using data visualization
* Identify relationships between temperature and other variables
* Build a machine learning model for temperature prediction
* Generate insights from weather data

 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

 Dataset

The dataset includes:

* Temperature
* Humidity
* Wind Speed
* Pressure
* Precipitation
* Cloud Cover
* Solar Radiation

 Project Workflow

 Data Preprocessing

* Removed missing values
* Converted categorical data into numerical form
* Handled outliers
* Standardized data

 Exploratory Data Analysis (EDA)

* Distribution analysis using histograms
* Relationship analysis using scatter plots
* Correlation analysis using heatmaps

 Feature Engineering

* Created temperature categories (Low, Medium, High)
* Created wind categories

 Model Building

* Applied Linear Regression
* Trained model using selected features

 Evaluation

* R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

 Key Insights

* Most observations fall under the medium temperature category
* Temperature varies with wind speed
* Weather variables show weak to moderate correlation
* Multiple factors influence temperature

 Visualizations

 Temperature Distribution

![Temperature](images/temp.png)

 Temperature vs Wind

![Temp vs Wind](images/temp_wind.png)

Correlation Heatmap

![Heatmap](images/heatmap.png)

 Category Distribution

![Category](images/pie.png)

 Comparative Analysis

![Comparison](images/grouped.png)

 How to Run

1. Clone the repository

```
git clone https://github.com/your-username/weather-data-analysis.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the project

```
python main.py
```

## Project Structure

```
weather-data-analysis/
│
├── data/
├── images/
├── main.py
├── requirements.txt
└── README.md
```

## Future Improvements

* Use advanced models (Random Forest, XGBoost)
* Add real-time data
* Build dashboard
* Improve prediction accuracy

## Author

Nandu
