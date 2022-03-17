# **Project Overview**

* Created an app that estimates Internship stipends on the basis of personalized factors for each individual user with inputs ranging widely from location, skills, perks, duration, etc.

* Scraped over 15000+ internship postings from Internshala.com using web scraper written on Python.

* Cleaned and manipulated the entire data extensively on Python to make it usable.

* Engineered new features and performed deep exploration on the data.
* Performed various pre-modelling statistical tests to understand the data better.

* Built a machine learning model with the best hyper-parameters.

* Built a client facing API using flask from scratch and hosted the app online for anyone to use.

**[Link of Productionized Model](https://flaskinternshalamodel-production.up.railway.app/)**

## Potential Users

* Students interested in knowing the benefits and the amount of stipend they should expect when applying for internships.

* Companies / Organizations / Firms while deciding how much they should offer as stipend when issuing new internship positions.

## Resources Used
* **Web Scraper:** [GitHub Link](https://github.com/het-parekh/Internshala-Web-Scraper-Internshala.com)
(The owner can be contacted on [hetparekh26@gmail.com](mailto:hetparekh26@gmail.com) for any inquiries.)
> With some modifications, tweaking and debugging, I was able to make the code work successfully.

## Features
The following features were retrieved with the Web Scraper:

- Title
- Company
- Location
- Duration
- Stipend
- Apply By
- Applicants
- Skills Required
- Perks
- Number of Openings
- Link

## Data Cleaning
Extensive changes were made to make the data usable for our project. Some of the important ones are mentioned below: 

 - Filtered our target variable on the basis of paid and unpaid stipend.
 - Made non-uniform Stipend and Duration data uniform, performed various transformations and conversions as and when necessary and finally parsed out only the numeric amount stipend per month and duration in number of months.
 - Performed hypothesis tests with One-Way ANOVA and t-tests to verify significances of observations in Location, Title, Applicants and made changes such as grouping, etc. as and when viable.
 - Used tests like Tukey's HSD, etc. to confirm our significance findings between pairs before making changes.
 - Filled missing values and parsed out Skills and Perks separated by commas into lists and proceeded to create dummy variables later with Multi Label Binarizer for each observation.

## Data Exploration
We explored the data from various angles to find all the interesting points of interest. Some of them are provided below:
* Top 15 Most Required Skills
![Top 15 Most Popular Skills](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/skills.png)

* Most Offered Perks
![Most Offered Perks](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/perks.png)

* Top 15 Hiring Companies
![Top 15 Hiring Companies](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/company.png)
* Top 15 Titles
![Top 15 Titles](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/title.png)
* Top 15 Locations offering most In-Office Internships
![Top 15 Locations offering most In-Office Internships](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/location.png)
* Ratio of In-Office to Work From Home
![Ratio of In-Office to Work From Home](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/wfhandothers.png)
* Duration - Stipend KDE Plot
![Duration - Stipend KDE Plot](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/stipenddurationkde.png)
* Number of Openings - Stipend KDE Plot
![Number of Openings - Stipend KDE Plot](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/stipendnoopenkde.png)

## Model Building
We transformed the target variable with Box-Cox Transformation after finding the optimal lambda value. We decided to use this transformation with minimal skewness of -0.02 for our Model Building. 
* Before and After Box-Cox Transformation
![Before and After Box-Cox Transformation](https://github.com/rishi5565/internshala-stipend-estimator/raw/main/EDA%20Images/boxcoxtrans.png)

We created dummy variables and performed various Pre-Modelling tests such as One-Way ANOVA, Pearson Correlation, Variance Inflation Factor(VIF), etc.

Next, we decided to use Light Gradient Boosting Machine (LGBM) Regressor to train our model.
**The reason being is that our Dataset is quite large (14,000+ observations) and Light GBM is very good at handling large size of data (above 10,000) while also staying efficient as it takes lower memory to run. It also performs better on Datasets with higher complexity as it provides options for regularization and can also be further hyper-parameter tuned to improve the balance between bias and variance.**

We tuned the hyper-parameters of our model to reduce the complexity and improve performance. We provided L2 Regularization with an ideal lambda value that we were able to derive by running a For-Loop. We used L2 Regularization because it disperses the error terms in all the weights which can lead to more accurate final model that can generalize better.

Next, we performed a Randomized Search with multiple fits to find the best hyper-parameters for our model.

### Model Performance
* Mean Absolute Error: ~ Rs. 3000

## Productionization
In this final step, we built a Flask API and hosted it online that takes in inputs from the user such as location, skills, perks, duration, etc. and returns an estimated stipend amount.

**[Link of Productionized Model](https://flaskinternshalamodel-production.up.railway.app/)**

### [Note]: ***Please refer to the Detailed Project Report word document for all the in-depth information regarding every decision made and the entire thinking process while working on this project. The above information is just a brief summary of the project.***

Thank You,
Rishiraj Chowdhury
[rishiraj5565@gmail.com](mailto:rishiraj5565@gmail.com)
