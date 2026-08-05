🔗 **Live Demo:** [houseprice.abhijeet.dpdns.org](http://houseprice.abhijeet.dpdns.org)

\# House Price Prediction — Data Analyst Portfolio Project



\## Problem Statement

Accurately estimating house sale prices helps real estate companies, buyers, and 

sellers make informed pricing decisions. This project builds a regression model 

to predict house sale prices based on property features, identifying which 

factors most strongly influence value.



\## Dataset

\- \*\*Source:\*\* \[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

\- \*\*Size:\*\* 1,458 houses (after outlier removal), 79 features

\- \*\*Target:\*\* `SalePrice` (log-transformed for modeling)



\## Approach

1\. Data cleaning (handled 19 columns with missing values, using context-aware strategies)

2\. Exploratory Data Analysis (target distribution, correlation analysis, outlier detection)

3\. Log transformation of target to correct right-skew

4\. Feature engineering (one-hot encoding 43 categorical columns)

5\. Model building — compared Linear Regression vs Random Forest

6\. Model evaluation — selected based on R² and RMSE



\## Key Findings

\- \*\*Overall Quality\*\* is by far the strongest price driver — more than 4x the influence of the next feature

\- \*\*Living area, garage size, and basement size\*\* form a strong second tier of price drivers

\- Two outlier sales (large homes, unusually low price) were identified and removed before modeling

\- Relationships in the data are largely linear once the target is log-transformed



!\[Sale Price Distribution](images/saleprice\_distribution.png)



!\[Log-Transformed Sale Price Distribution](images/saleprice\_log\_distribution.png)



!\[Price by Quality and Living Area](images/price\_by\_quality\_and\_area.png)



\## Model Performance



| Metric | Linear Regression | Random Forest |

|---|---|---|

| R² Score | \*\*0.888\*\* | 0.872 |

| RMSE (log scale) | 0.137 | 0.147 |

| RMSE (actual $) | \*\*$21,891\*\* | $24,391 |


\*\*Final model: Linear Regression\*\* — selected for higher R² and lower error. 

The dataset's relationships proved largely linear once properly transformed and encoded.



!\[Feature Importance](images/feature\_importance.png)



\## Business Recommendation

Prioritize quality-related renovations (materials, finishes) over square footage 

expansion alone for the best return on investment. Use quality-tier-adjusted 

comparisons rather than flat per-square-foot pricing for more accurate listings.



\## Tech Stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn



\## How to Run

1\. Clone this repo

2\. Install dependencies: `pip install -r requirements.txt`

3\. Open `notebooks/house\_price\_prediction.ipynb` and run all cells

