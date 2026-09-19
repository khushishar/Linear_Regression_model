# Linear_Regression
This is a notebook predicting house prices using linear regression made from scratch. 




1.Data Overview
Dataset: The dataset chosen is a housing dataset (Housing.csv) sourced from Kaggle (https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset)

What are we trying to predict?
The objective of this project is to predict the price of a house based on its characteristics. The dataset contains information about the number of bedrooms, bathrooms, living area size, lot size, and location details. The target variable is Price, which represents the price of the house. Since there is one clear  continuous numerical and  all the other predictors are contributing to target variance, this problem can be formulated as a regression problem. 

Dependent and Independent Variables Dependent variable (Target):  
Price : the house price that we want to predict. 
Independent variables: 
bedrooms : Number of bedrooms 
bathrooms : Number of  bathrooms
sqft_living : Living area size in square feet
floors : Number of floors 
view: Quality level of property view (0 to 4)
grade : Overall grade rating (1 to 13)
sqft_above : Living area above ground level in square feet
zipcode: Property location zip code
lat: Latitude coordinate of property location
sqft_living15 : Living area size of 15 nearest properties in square feet

2. Dataset Statistics and Visualization 

Before preprocessing the dataset had the following properties:
• Number of Observations: The dataset contains exactly 21,613 unique records
• Number of Features: There are 21 features (columns) in total, which include 10 predictive       variables and 1 target variable.
 • Data Types:
Numerical (float64,int64) : Four variables are numerical, allowing for continuous mathematical calculations.(Sqft_living,sqft_above,lat,sqft_living15)
Categorical (object): Six variables are categorical (bedrooms,bathrooms,floors,view,grade,zipcode)
• Missing Values: There are zero missing values across all columns. (The dataset is completely clean, so no data imputation techniques are required. )
• Duplicate Records: The dataset contains 0 duplicate records, ensuring that every observation represents a distinct entry and will not artificially bias a machine learning model.

Target Variable Statistics: The target variable is Price. It has a mean (average) value of approximately 540,088.58, with a minimum price of 75,000 and a maximum price of 7,700,000. The standard deviation is 367,126.83, indicating a vast difference, and a wide distribution of housing values across the dataset.

Visualization: 


<img width="1340" height="525" alt="price_histogram1plot" src="https://github.com/user-attachments/assets/acaa19bd-2e86-4907-a123-2f59659181b6" />


The histogram shows that house prices  follow a highly right skewed bell-shaped Normal Distribution.The highest concentration of properties is under 1 million (1M). The first bar shows over 12,000 homes in the lowest price bracket, followed by nearly 8,000 homes in the next tier.
As price increases past 1.5M, the number of houses drops drastically.Moreover,there is a very thin, long tail stretching all the way out to 8 million, indicating some outlier properties.

<img width="1340" height="525" alt="scatterplot1" src="https://github.com/user-attachments/assets/093cfe13-e5e4-42f3-bda2-fcb1e45139ec" />



The scatter plot between the bedrooms and bathrooms reveals a general trend of  positive correlation between bedrooms and bathrooms for standard homes. As the number of bedrooms grows from 1 to 6, the number of bathrooms typically increases as well.
The most expensive properties (yellow and orange dots) are clustered around 5 to 6 bedrooms and 6 to 8 bathrooms.The 33-Bedroom outlier is a bizarre outlier with 33 bedrooms but less than 2 bathrooms. 
The dense cluster of dark blue dots under 4 bedrooms and 3 bathrooms shows where the vast majority of the housing market sits financially.

<img width="1340" height="525" alt="scatter plot2" src="https://github.com/user-attachments/assets/6400c3a0-129c-40b6-9fdd-402727a1c9c8" />

The scatter plot between the zip codes and price reveals zip codes which are  more expensive, or affordable.The isolated dots at the very top of certain columns represent extreme outliers. The solid, thick blue lines at the bottom of each column show where the vast majority of homes are sold. Most homes across almost all these zip codes are clustered under the $1 Million to $2 Million mark.More affordable areas zipcodes like 98168, 98023, or 98070 have much shorter columns that top out well below $2 Million, indicating areas where housing prices are generally lower and more tightly grouped.


<img width="1340" height="525" alt="s3" src="https://github.com/user-attachments/assets/3fd49ba4-5042-4537-88b0-4dee5b97b5ee" />



 The scatter plot between the square-feet area of the living space and the price reveals a strong positive correlation, as you move from left to right (larger house size), the dots move upward (higher price). This means that larger homes generally cost more money, which matches typical real estate expectations.
The data starts out very tightly packed at the lower-left corner (small homes) and spreads out wide like a cone or funnel as homes get larger.
At the top right, you can see the absolute highest-priced homes (around $7 Million to $8 Million) are also among the largest (10k to 12k sq ft).There is a notable anomalous outlier on the far right: a massive home around 13.5k sq ft that sold for a relatively low price (under $2.5 Million). This could indicate a property that needs massive renovations, is located in a much cheaper rural zip code, or represents a data entry error. 


<img width="761" height="605" alt="download (2)" src="https://github.com/user-attachments/assets/0a3e3ae1-44c9-4a6e-9e3a-248d8b113ec6" />

The correlation matrix heatmap shows the relationships between different numerical features The numbers range from 0 to 1.0 (representing positive correlation), where a higher number means two variables tend to increase together. 
Sqft_living has the strongest correlation with price (0.70). This confirms what we saw in your previous scatter plot,as living space size increases, the price increases significantly.
Bathrooms have a moderate-to-strong correlation with price (0.53). More bathrooms generally mean a higher property value.
sqft_lot has almost zero correlation with price (0.09). Surprisingly, the overall size of the land/yard has very little impact on the final sale price in this dataset.
The strongest Internal Relationship:sqft_living and bathrooms have a very strong correlation of 0.75. This makes practical sense, as larger houses naturally tend to be built with more bathrooms.Other Interesting Notes:bedrooms vs. bathrooms (0.52): There is a moderate relationship here, but it's weaker than sqft_living vs. bathrooms.sqft_basement (0.44 with sqft_living, 0.32 with price): A larger basement contributes moderately to both the total living space and the overall value of the home.

 



3. Data Preprocessing 
• Dealing with outliers:
The IQR method was used, it is a non-parametric approach used to detect anomalies based on the distribution of data points. It is highly robust because it relies on percentiles, meaning it is not heavily influenced by the very outliers it is trying to detect. It cleans the dataset to prevent extreme, unrepresentative properties (like a $12M mega-mansion or a house with 11 bedrooms) from heavily skewing predictive machine learning models or inflating statistical averages. 
This changes the number of observations from 21,613 to 19,457.

 After removing outliers:

<img width="1340" height="525" alt="after_outliers" src="https://github.com/user-attachments/assets/0395fcc5-f6f6-4b81-b162-417e17d6671b" />


 Looking at box plot:

 <img width="1340" height="525" alt="newplot" src="https://github.com/user-attachments/assets/e3d038b9-dc3a-4110-b6c3-d6e71b557aad" />



The horizontal line inside the purple box shows that the median house price is roughly $430,000. Half of the homes sold for more than this, and half sold for less
The box represents the middle 50% of the dataset (the Interquartile Range).The bottom edge (Q1) is around $310,000.The top edge (Q3) is around $590,000.This means 50% of all homes in the cleaned dataset cost between $310k and $590k.
The lower whisker stops around $80,000.The upper whisker stops right around $1 Million.

Outliers Removed: The distinct absence of scattered individual dots beyond the top horizontal whisker.  
Both the plots show the distribution of house prices after applying your IQR cleaning method.Comparing this to previous scatter plots where prices went all the way up to $8 Million,  the cleaning method successfully removed those extreme values, narrowing the price scale to under $1.1 Million.

<img width="1340" height="525" alt="newplot (1)" src="https://github.com/user-attachments/assets/4ef0505c-fa00-4ec0-acd0-539a24274015" />


The IQR outlier filter perfectly at work here. The maximum price on the color bar scale caps out right around $1.1 Million, matching the exact ceiling established in the previous box plot. 

• Data encoding:
As the dataset has two major categorical features i.e. view and zipcode if as integers, the linear regression model would assume zip code 98102 is statistically "greater" or more valuable than 98004( similarly for view).This lead to finding different encoding techniques after experimenting with Target encoding, Binary Encoding and One-Hot Encoding. 
We started by establishing that encoding must happen after the train-test split to prevent data leakage and avoid fake, inflated accuracy scores. The R² score dropped from 0.8 to 0.4 when switching from One-Hot to Binary Encoding. Concluding  that One-Hot Encoding is the best choice for linear models because it keeps categories independent and avoids breaking linear assumptions. 

• Feature Engineering:
There were few features which didn’t show good correlations with the target. They were dropped completely. 
The columns dropped are 
'id','long','sqft_lot','yr_built','yr_renovated','sqft_lot15','date','sqft_basement','waterfront'. 
This has helped improve the efficiency of the regression.

<img width="861" height="693" alt="download" src="https://github.com/user-attachments/assets/5800d23b-b6f0-44fe-a220-8135f1b200ac" />

The heatmap below shows the relations of each feature with price. Looking at this the columns were dropped, any feature which has <0.2 correlation were dropped. Also, it's understood that -ve correlation is considered -coeff.

 •  Feature Scaling / Normalization: 
The features are normalised as it-
Balances Feature Importance: As the prices are in the hundreds of thousands (e.g., $500,000) and encoded variables are just 0 or 1, the massive scale of the price column can destabilize the mathematical solver of the linear regression model.
Matches Linear Assumptions: Linear regression models perform best, converge faster, and provide highly stable coefficients when numerical features are normally distributed and scaled around zero.
Mean and standard deviation of each feature in the training data is calculated and then z score of reach datapoint is calculated. z= pt-mean/std

Why do we use training  data mean only?
Prevent Data Leakage: Calculating the mean of the test data (or the entire dataset combined), information from the test set sneaks into the training process. This makes your model look more accurate than it actually is.
Simulate Real-World Scenarios: In production, the model receives brand-new, unseen data one piece at a time. You cannot calculate the mean of future data you have not seen yet.
Maintain Consistency: The model learns patterns based on the scale of the training data. Shifting the scale using the test set's own mean, the same input value will mean something completely different to the model, leading to bad predictions.

 • Training and Testing data split: 
The dataset was divided using a standard 80% training split and 20% testing split. From the 19,457  total observations, 15,565  records were allocated to train the machine learning algorithms, while the remaining 3,891 records were completely held back to serve as an unseen testing pool for final evaluation. 

 Linear Regression Least Squares Method: 
The goal of linear regression is to find the optimal weights that minimize the distance between the model’s predictions and the actual house prices. 
• The Implementation 
Adding the Bias Term (Intercept): A matrix column filled entirely with 1s is added to the scaled training features X. This allows the model to calculate θ0 which represents the baseline house price when all other features are zero.
 Matrix Transpose and Multiplication: We compute the transpose of our feature matrix X.T and multiply it by the original matrix X.
Pseudo-Inverse Calculation: We compute the Moore-Penrose pseudo-inverse (np.linalg.pinv) of that result to ensure numerical stability, even if features are highly correlated.
Final Weight Calculation: The inverted matrix is multiplied back by the transposed features and the target house prices Y to isolate the final θ parameters. 

What These Parameters Mean (Interpretation): 
Baseline Value Θ0:The intercept shows that an area with completely average metrics across the board has a base estimated house price of 470,190.86. (This matches our dataset’s median price(4,30,000)because the data was standardized around 0).
Most Critical Feature θ1: sqft_living  has the largest weight. For every standard deviation increase in an area’s income, the house price jumps by 58,187.
Least Critical Feature θ86: Floor 2  has the smallest weight, meaning holding all other numerical and categorical features constant a house being on 2nd reduces the predicted price more than any other individual characteristic. Also, a house located on the 2nd floor costs 13,806.72 less than an otherwise identical house located in the baseline.

Linear Regression using Gradient Descent: 
Unlike the Normal Equation, which solves for the optimal parameters analytically in a single step, Gradient Descent is an iterative optimization algorithm. It starts with initial parameters set to zero and gradually adjusts them to minimize the Mean Squared Error (Cost Function) by stepping in the opposite direction of the gradient. 
Effect of Learning Rate α on Convergence:
 The learning rate α determines the size of the steps the algorithm takes toward the minimum cost. The three tested learning rates show distinctly different behaviors: 
Small Learning Rate α= 0.001: As shown by the blue line, convergence is extremely slow. Because the step size is too small, the cost decreases gradually but fails to reach the absolute minimum within 1,000 iterations. This is confirmed by its final parameters which are still far away from the true optimal values. 
Moderate Learning Rate α= 0.01: The orange line demonstrates a steady, healthy convergence. The cost drops rapidly within the first 200 iterations and levels off completely near zero around iteration 400. Its learned parameters are incredibly close to the true analytical solution. 
Large Learning Rate α= 0.1: The green line represents optimal and rapid convergence. The step size is large enough to drop the cost function directly to its minimum in fewer than 50 iterations without overshooting. Its final learned parameters perfectly match the exact mathematical values calculated by the Normal Equation scratch implementation down to the decimal point. 

<img width="846" height="547" alt="download (1)" src="https://github.com/user-attachments/assets/8ba83b47-2e36-4146-8d5e-7b019d61998d" />


The graph illustrates how changing the learning rate directly controls the speed at which the cost function approaches zero. A small rate of α= 0.001 creates a slow, steady descent that requires far more than 1,000 steps to finish training. Increasing the rate to α= 0.01 or α= 0.1 drastically improves efficiency, allowing the model to reach full stability and find the absolute optimal weights within 400 and 50 iterations, respectively.

Comparison of the models :
Optimization Method: Normal Equation uses exact analytical calculation; Gradient Descent uses iterative, step-by-step optimization.
Iterations Required: Normal Equation requires exactly one calculation; Gradient Descent needs roughly 50 iterations to settle. 
Scalability Limitations: Normal Equation shows poor scalability with huge feature counts; Gradient Descent scales excellently for large datasets. 

Scaling Sensitivity: Normal Equation works directly with raw numbers; Gradient Descent highly requires feature scaling to function properly. 
Calculated Evaluation Metrics:  
 MAE  : 61341.49
Mean Absolute Error (MAE) measures the average absolute size of the mistakes made by our model. On average, the model's predicted house prices miss the actual market sales prices by 61341.49

MSE  : 7052068949.31 and RMSE : 83976.60
Mean Squared Error (MSE) squares each error before taking the average, resulting in a very large value that is physically difficult to interpret because the units are "squared dollars." & Root Mean Squared Error (RMSE)  resolves this by taking the square root of the MSE, bringing the units back to standard dollars. 
Looking at the average error rate of 17.82%  the model is doing fairly well. Also, as RMSE < Standard deviation of house price it suggests the model doing better than baseline i.e. just taking the mean of the values.

R²   : 0.8278
Coefficient of Determination (R² Score) tells us what percentage of the variance in housing prices can be explained by the model's input features. The model has achieved an R² score of 0.8278. This means that 82.78% of the differences in house prices are directly explained by the features of the data set.
