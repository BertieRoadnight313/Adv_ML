import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('Week 4/Online_Retail.csv')
print(data.head())
print(data.info())

print(data.columns)
print(data.Country.unique())

data['Description'] = data['Description'].str.strip()

data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

data['InvoiceNo'] = data['InvoiceNo'].astype('str')

data = data[~data['InvoiceNo'].str.contains('C')]

# Split data by region

basket_France = (data[data['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('InvoiceNo'))
print("France Shape", basket_France.shape)


basket_UK = (data[data['Country'] == "United Kingdom"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('InvoiceNo'))
print("UK Shape", basket_UK.shape)


basket_Portugal = (data[data['Country'] == "Portugal"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('InvoiceNo'))
print("Portugal Shape", basket_Portugal.shape)


basket_Sweden = (data[data['Country'] == "Sweden"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('InvoiceNo'))
print("Sweden Shape", basket_Sweden.shape)

# Hot encoding the datasets

def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_encoded_France = basket_France.map(hot_encode)
basket_encoded_UK = basket_UK.map(hot_encode)
basket_encoded_Portugal = basket_Portugal.map(hot_encode)
basket_encoded_Sweden = basket_Sweden.map(hot_encode)

basket_France = basket_encoded_France
basket_UK = basket_encoded_UK   
basket_Portugal = basket_encoded_Portugal
basket_Sweden = basket_encoded_Sweden

# Building the model for France

frq_items_France = apriori(basket_France, min_support = 0.05, use_colnames = True)
rules_FR = association_rules(frq_items_France, metric = "lift", min_threshold = 1)

rules_FR = rules_FR.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules_FR.head(10))