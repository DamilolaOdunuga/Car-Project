#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:28:05 2024

@author: damilolaodunuga
"""

# Import packages need for analysis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Import dataset
cars = pd.read_csv("~/Documents/Documents - MacBook Air/Personal Projects_2/cars.csv")

# Exploratory Analysis
print(cars.head(5))
print(cars.describe())
cars.info()

# Finding nulls
print(cars.isnull().sum())
print(cars.isnull().any())
print(cars.dtypes)

# Find duplicate data
print(cars[cars.duplicated()])

### DATA CLEANING
## TREATING MISSING VALUES OR NULLS BY REPLACING WITH MEAN OR MODE OF EACH COLUMN
# FINDING MEAN

mean_mileage = cars['mileage'].mean()
mode_engine = cars['engine'].value_counts().idxmax()
mode_trans = cars['transmission'].value_counts().idxmax()
mode_drivetrain = cars['drivetrain'].value_counts().idxmax()
mode_fueltype = cars['fuel_type'].value_counts().idxmax()
mode_mpg = cars['mpg'].value_counts().idxmax()
mode_extr = cars['exterior_color'].value_counts().idxmax()
mode_intr = cars['interior_color'].value_counts().idxmax()
mean_accident = cars['accidents_or_damage'].mean()
mean_owner = cars['one_owner'].mean()
mean_personal = cars['personal_use_only'].mean()
mode_sellname = cars['seller_name'].value_counts().idxmax()
mean_sellrate = cars['seller_rating'].mean()
mean_driverate = cars['driver_rating'].mean()
mean_price = cars['price_drop'].mean()


# REPLACING NULLS WITH MEAN/MODE
cars['mileage'] = cars['mileage'].fillna(mean_mileage)
cars['engine'] = cars['engine'].fillna(mode_engine)
cars['transmission'] = cars['transmission'].fillna(mode_trans)
cars['drivetrain'] = cars['drivetrain'].fillna(mode_drivetrain)
cars['fuel_type'] = cars['fuel_type'].fillna(mode_fueltype)
cars['mpg'] = cars['mpg'].fillna(mode_mpg)
cars['exterior_color'] = cars['exterior_color'].fillna(mode_extr)
cars['interior_color'] = cars['interior_color'].fillna(mode_intr)
cars['accidents_or_damage'] = cars['accidents_or_damage'].fillna(mean_accident)
cars['one_owner'] = cars['one_owner'].fillna(mean_owner)
cars['personal_use_only'] = cars['personal_use_only'].fillna(mean_personal)
cars['seller_name'] = cars['seller_name'].fillna(mode_sellname)
cars['seller_rating'] = cars['seller_rating'].fillna(mean_sellrate)
cars['driver_rating'] = cars['driver_rating'].fillna(mean_driverate)
cars['price_drop'] = cars['price_drop'].fillna(mean_price)

## VERIFY NULL HAS BEEN TREATED
print(cars.isnull().any())
print(cars.isnull().sum())

# CONVERT DATA TYPES FROM FLOAT TO INTEGER
cars['mileage'] = cars['mileage'].round().astype(int)
cars['accidents_or_damage'] = cars['accidents_or_damage'].round().astype(int)
cars['one_owner'] = cars['one_owner'].round().astype(int)
cars['personal_use_only'] = cars['personal_use_only'].round().astype(int)
cars['seller_rating'] = cars['seller_rating'].round().astype(int)
cars['driver_rating'] = cars['driver_rating'].round().astype(int)
cars['driver_reviews_num'] = cars['driver_reviews_num'].round().astype(int)
cars['price_drop'] = cars['price_drop'].round().astype(int)
cars['price'] = cars['price'].round().astype(int)

# VISUALIZATION BEFORE HANDLING MISSING VALUE
# Histogram plot of mileage column
miles = cars['mileage']
plt.ticklabel_format(style='plain') # remove scientific notation
plt.title('Car Miles')
miles.plot()

# Scatter plot of mileage and price
sns.scatterplot(data=cars, x='mileage', y='price')
plt.title('Mileage vs Price')
plt.show()

print(cars.describe())

# STATISTICAL ANALYSIS
# Average price by year
average_price_by_year = cars.groupby('year')['price'].mean()
print(average_price_by_year) 

# Average price drop by year
average_price_drop_by_year = cars.groupby('year')['price_drop'].mean()
print(average_price_drop_by_year)

# Most car model sold by year
model_by_year = cars.groupby('year')['model'].agg(pd.Series.mode)
print(model_by_year)

#Correlation of milage and price drop
corr_mile_by_price_drop = cars['mileage'].corr(cars['price_drop'])
print(f"Correlation between mileage and price drop: {corr_mile_by_price_drop}") 

# Correlation of mileage and price
corr_mile_by_price = cars['mileage'].corr(cars['price'])
print(f"Correlation between mileage and price: {corr_mile_by_price}")

# Create dataset for correlation
cars_data = cars[['mileage', 'accidents_or_damage', 'one_owner', 'personal_use_only', 
                  'seller_rating', 'driver_rating', 'driver_reviews_num', 'price']]

# Calculate the correlation matrix
correlation_matrix = cars_data.corr()

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Car manufacturere in each year
manufacturer_by_year = cars.groupby('year')['manufacturer'].agg(pd.Series.mode)
print(manufacturer_by_year)