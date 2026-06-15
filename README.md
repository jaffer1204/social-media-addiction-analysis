# Social Media Addiction Analytics and Insights

## Project Overview

This project analyzes student social media usage patterns and studies how different factors such as daily usage hours, sleep duration, academic performance, mental health score, and platform preference relate to social media addiction. The project uses Python for data analysis, visualization, and machine learning to predict the addiction score of students based on the available dataset.

The project is suitable for academic learning and college internship submission because it covers the complete workflow of a data analysis and machine learning task, from loading the dataset to building and evaluating a prediction model.

## Objectives

- To understand the social media usage behavior of students.
- To identify the most commonly used social media platforms.
- To analyze the relationship between social media usage, sleep hours, mental health, and academic performance.
- To visualize important trends and correlations in the dataset.
- To build a machine learning model that predicts student social media addiction score.
- To evaluate the model performance using accuracy.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest Classifier

## Dataset Description

The dataset used in this project is stored in:

```text
dataset/Students Social Media Addiction.csv
```

The dataset contains 705 student records with features related to social media behavior and student lifestyle.

Important columns include:

- `Student_ID`: Unique student identifier
- `Age`: Age of the student
- `Gender`: Gender of the student
- `Academic_Level`: Academic level of the student
- `Country`: Student's country
- `Avg_Daily_Usage_Hours`: Average daily social media usage in hours
- `Most_Used_Platform`: Social media platform used most often
- `Affects_Academic_Performance`: Whether social media affects academic performance
- `Sleep_Hours_Per_Night`: Average sleep hours per night
- `Mental_Health_Score`: Mental health score of the student
- `Relationship_Status`: Relationship status of the student
- `Conflicts_Over_Social_Media`: Number of conflicts related to social media
- `Addicted_Score`: Target variable representing the level of social media addiction

## Project Workflow

1. Load the dataset using Pandas.
2. Display the first few rows of the dataset.
3. Check dataset information, missing values, and duplicate records.
4. Clean the dataset by removing duplicate rows.
5. Perform exploratory data analysis using charts and graphs.
6. Encode categorical columns using `LabelEncoder`.
7. Split the dataset into training and testing data.
8. Train a Random Forest Classifier model.
9. Predict addiction scores on the test data.
10. Evaluate the model using accuracy score.

## Visualizations

The project includes the following visualizations using Matplotlib and Seaborn:

- Most used social media platforms
- Distribution of average daily usage hours
- Relationship between sleep hours and social media usage
- Correlation heatmap of numerical features

These visualizations help in understanding usage trends and identifying relationships between different factors in the dataset.

## Machine Learning Model

The project uses a Random Forest Classifier for prediction.

Model steps:

- Categorical text columns are converted into numeric values using `LabelEncoder`.
- `Addicted_Score` is selected as the target variable.
- All other columns are used as input features.
- The dataset is split into 80% training data and 20% testing data.
- The Random Forest model is trained on the training data.
- Predictions are made on the test data.
- Model performance is evaluated using accuracy score.

## Accuracy

The model achieved the following accuracy:

```text
Model Accuracy: 0.9858156028368794
```

This is approximately:

```text
98.58%
```

The accuracy value may slightly vary when the model is run again because the Random Forest model is created without a fixed `random_state`.

## How to Run the Project

Install the required Python libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Run the dataset inspection and cleaning script:

```bash
python main.py
```

Run the visualization script:

```bash
python analysis.py
```

Run the machine learning model:

```bash
python model.py
```

On Windows, if `python` is not available, use:

```bash
py model.py
```

## Project Files

```text
Social media/
+-- dataset/
|   +-- Students Social Media Addiction.csv
+-- main.py
+-- analysis.py
+-- model.py
+-- README.md
```

## Conclusion

This project successfully analyzes student social media addiction using Python. The analysis shows how factors such as daily usage hours, sleep hours, mental health score, academic performance, and conflicts over social media can be used to study addiction patterns. The Random Forest Classifier achieved high accuracy, showing that machine learning can be useful for predicting addiction scores from student behavior data.

## Future Improvements

- Add more machine learning models for comparison, such as Logistic Regression, Decision Tree, and Support Vector Machine.
- Use cross-validation for more reliable model evaluation.
- Add confusion matrix and classification report for deeper performance analysis.
- Save generated visualizations as image files.
- Build a simple web application using Streamlit or Flask.
- Improve preprocessing by using separate encoders for different categorical columns.
- Tune Random Forest hyperparameters to improve model stability and performance.
