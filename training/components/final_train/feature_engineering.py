import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from datetime import datetime

df = pd.read_excel('recomendation.xlsx')

df=df.drop(columns=['PurchaseHistory','CartActivity','WishlistActivity','AbandonedCartData','BrowsingHistory',"ProductName"])

pd.set_option("display.max_columns",None)
df.sample(2)

# Extract unique ItemIDs for random sampling
unique_item_ids = df['ItemID'].unique()

# Function to assign a random list of ItemIDs
def assign_random_items(item_list, n=3):
    return list(np.random.choice(item_list, size=n, replace=False))

# Create new columns with random ItemIDs

df['CartActivity'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
df['WishlistActivity'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
df['AbandonedCartData'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
df['BrowsingHistory'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))

# Display the first few rows to verify
df[['UserID', 'CartActivity', 'WishlistActivity', 'AbandonedCartData', 'BrowsingHistory']].head()

# Group by 'UserID' and collect all 'ItemID's for each user as a list
purchase_history_df = df.groupby('UserID')['ItemID'].apply(list).reset_index()

# Rename the column to 'PurchaseHistory' for clarity
purchase_history_df.rename(columns={'ItemID': 'PurchaseHistory'}, inplace=True)

# Merge this back to the original DataFrame if you need to retain all original data with the new feature
df = df.merge(purchase_history_df, on='UserID', how='left')

# Display the final DataFrame with the new 'PurchaseHistory' column
df.head()

# According to our Literature Review we do not need of Purchase date so we can decidede to drop this feature
df.drop('PurchaseDate', axis=1, inplace=True)


df.isnull().sum()

# we can fill the nan values
df['Size'] = df['Size'].apply(lambda x: np.random.choice(df['Size'].dropna()) if pd.isna(x) else x)


df.isnull().sum()

label_encoders = {}
for col in ['Gender','Location','MembershipLevel','Occupation','DeviceType','Category','Brand','Color','Size']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

date_columns = ['SignUpDate','ReleaseDate']

for col in date_columns:
    df[col + '_Year'] = df[col].dt.year
    df[col + '_Month'] = df[col].dt.month
    df[col + '_Day'] = df[col].dt.day
    df[col + '_DayOfWeek'] = df[col].dt.dayofweek

df.drop(columns = date_columns,inplace = True)

df.head(1)

scaler = MinMaxScaler()
numerical_columns = ['Clicks', 'Views', 'TimeSpentOnItem', 'SessionDuration', 'Age', 'Price', 'Discount', 'Stock', 'Ratings', 'PopularityScore']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


scaler = MinMaxScaler()
numerical_columns = ['Clicks','Views','TimeSpentOnItem','SessionDuration','Age','Price','Discount','Stock','Ratings','PopularityScore']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

df.sample()

df[['CartActivity','WishlistActivity','AbandonedCartData','BrowsingHistory']].head(3)

df.to_csv("new_df")

df.sample(1)

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize Tags
tfidf_tags = TfidfVectorizer(max_features=100)
tags_matrix = tfidf_tags.fit_transform(df['Tags']).toarray()

# Add vectorized tags to the Dataframe
for i in range(tags_matrix.shape[1]):
    df[f'Tag_{i}'] = tags_matrix[:,i]

# Vectorize Description
tfidf_description = TfidfVectorizer(max_features=200)
description_matrix = tfidf_description.fit_transform(df['Description']).toarray()

# Add vectorized description to the Dataframe
for i in range(description_matrix.shape[1]):
    df[f'Description_{i}'] = description_matrix[:,i]

# Drop original text columns if not needed
df.drop(columns=['Tags','Description'],inplace=True)

df.sample()

df.shape

pip install textblob


from textblob import TextBlob

# Define a function to get sentiment polarity
def sentiment_analysis_function(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity


# Apply the sentiment function to each review
df['Reviews_Sentiment'] = df['Reviews'].apply(lambda x: sentiment_analysis_function(x) if pd.notnull(x) else 0)

# Drop the original Reviews column
df.drop(columns=['Reviews'], inplace=True)


df.sample(1)

df[['Income','Device','TimeOfInteraction']]

# Ordinal encoding for 'Income'
income_order = {'Low': 1, 'Medium': 2, 'High': 3}
df['Income'] = df['Income'].map(income_order)

# Perform one-hot encoding on 'Device' and 'TimeOfInteraction'
df = pd.get_dummies(df, columns=['Device', 'TimeOfInteraction'], prefix=['Device', 'Time'])



df.sample(1)

df[['Device_Desktop','Device_Mobile','Device_Tablet','Time_Afternoon','Time_Evening','Time_Morning','Time_Night']]

# List of columns you want to convert
columns_to_convert = ['Device_Desktop', 'Device_Mobile', 'Device_Tablet', 
                      'Time_Afternoon', 'Time_Evening', 'Time_Morning', 'Time_Night']

# Convert only the selected columns from boolean to numeric (True -> 1, False -> 0)
df[columns_to_convert] = df[columns_to_convert].astype(int)


df[['Device_Desktop','Device_Mobile','Device_Tablet','Time_Afternoon','Time_Evening','Time_Morning','Time_Night']]

df.sample(1)

df.to_excel('Preprocessed_recommendation.xlsx', index=False)

