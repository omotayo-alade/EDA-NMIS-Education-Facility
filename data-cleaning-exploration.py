# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Load data into data frame
df = pd.read_csv('data_raw.csv')
df.head()

df.shape
df.columns

# Drop irrelevant columns
df.drop(
    ['education_type', 'community', 'ward', 'facility_type_display', 'formhub_photo_id', 'gps', 'survey_id', 'latitude',
     'longitude', 'date_of_survey', 'sector'], axis=1, inplace=True)

# Define a new order of columns
order = ['facility_id', 'facility_name',
         'management', 'unique_lga',
         'improved_water_supply', 'improved_sanitation', 'phcn_electricity',
         'chalkboard_each_classroom_yn', 'num_classrms_total', 'num_toilets_total',
         'num_tchrs_with_nce', 'num_tchr_full_time', 'num_students_male',
         'num_students_female', 'num_tchrs_male', 'num_tchrs_female', 'num_students_total']

# Re-order columns
df = df[order]

# Rename Columns
df.columns = ['Facility_ID', 'Facility_name', 'Management', 'State',
              'Improved_water_supply', 'Improved_sanitation', 'Public_electricity',
              'Chalkboard/classroom', 'Classrooms', 'Toilets',
              'NCE_teachers', 'Fulltime_teachers', 'Male_students',
              'Female_students', 'Male_teachers',
              'Female_teachers', 'Total_students']

df.head()

# Check if there are missing values
df.isnull().any()

# Check the total number of missing values for each column
df.isnull().sum()

# Check dataset shape to decide whether to drop missing values on rows or column
df.shape

# Since dataset appears to have almost 99,000 rows, drop rows with missing values and retain all attributes(columns)
df.dropna(axis=0, inplace=True)

# Check to see if there are still missing values
df.isnull().sum().any()

# Check the new shape of data
df.shape

df.head()

states = ["Abia", "Adamawa", "Akwa_Ibom", "Anambra", "Bauchi", "Bayelsa",
          "Benue", "Borno", "Cross_River", "Delta", "Ebonyi", "Edo",
          "Ekiti", "Enugu", "fct", "Gombe", "Imo", "Jigawa", "Kaduna",
          "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
          "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
          "Sokoto", "Taraba", "Yobe", "Zamfara"]

# Join L.G.A(s) in State column into 37 unique values to represent 36 states in Nigeria plus FCT
for i in states:
    df.loc[df['State'].str.contains(i, case=False), 'State'] = i
    df.loc[df['State'].str.contains('fct', case=False), 'State'] = 'FCT - Abuja'
    df.loc[df['State'].str.contains('Cross_River', case=False), 'State'] = 'Cross River'
    df.loc[df['State'].str.contains('Akwa_Ibom', case=False), 'State'] = 'Akwa Ibom'

# Check the unique values
df['State'].unique()

# Check the number of unique values
df['State'].nunique()

# Check for duplicates
df.set_index('Facility_ID', inplace=True)
df.duplicated().value_counts()

# Drop duplicates
df.drop_duplicates(keep='first', inplace=True)

# Check if there's still any duplicate
df.duplicated().any()

df.head()

# Rename booleans to Yes and No to avoid issues
df[['Improved_water_supply', 'Improved_sanitation', 'Public_electricity', 'Chalkboard/classroom']] = \
    df[['Improved_water_supply', 'Improved_sanitation', 'Public_electricity', 'Chalkboard/classroom']].replace(
        [True, False], ['Yes', 'No'])

# Check categorical variables for unique values
unique = ['Management', 'Improved_water_supply', 'Improved_sanitation', 'Public_electricity', 'Chalkboard/classroom']
for i in unique:
    print(i, df[i].unique())

# Rename faith_based strings in Management column to 'private'
df.loc[df['Management'].str.contains('faith'), 'Management'] = 'private'

# drop rows containing 'none' in Management column
df.reset_index(inplace=True)
none = df['Management'] == 'none'
none_id = df[none]['Facility_ID'].tolist()
df.set_index('Facility_ID', inplace=True)
df.drop(index=none_id, inplace=True)
df.reset_index(inplace=True)

# Check for changes
unique = ['Management', 'Improved_water_supply', 'Improved_sanitation', 'Public_electricity', 'Chalkboard/classroom']
for i in unique:
    print(i, df[i].unique())

# Check data frame
df.head()

# Export cleaned data to csv
df.to_csv('data_clean.csv')

# Data Visualization

# Descriptive Statistics
df.describe()

# Univariate Distribution of Numerical variables
df_dist = df.select_dtypes(include=['float'])
df_dist.hist(bins=50, figsize=(15,13));
plt.savefig('univariate-distribution.png')

# Univariate Analysis of all categorical variables
fig = plt.figure(figsize=(13,15))
ax0 = fig.add_subplot(3,2,1)
ax1 = fig.add_subplot(3,2,2)
ax2 = fig.add_subplot(3,2,3)
ax3 = fig.add_subplot(3,2,4)
ax4 = fig.add_subplot(3,2,5)

sns.barplot(df['Management'].unique(), df['Management'].value_counts(), ax=ax0);
sns.barplot(df['Improved_water_supply'].unique(), df['Improved_water_supply'].value_counts(), ax=ax1);
sns.barplot(df['Improved_sanitation'].unique(), df['Improved_sanitation'].value_counts(), ax=ax2);
sns.barplot(df['Public_electricity'].unique(), df['Public_electricity'].value_counts(), ax=ax3);
sns.barplot(df['Chalkboard/classroom'].unique(), df['Chalkboard/classroom'].value_counts(), ax=ax4);

ax0.set_title('Private and Public Schools'), ax0.set_ylabel('Count')
ax1.set_title('Improved Water Supply'), ax1.set_ylabel('Count')
ax2.set_title('Improved Sanitation'), ax2.set_ylabel('Count')
ax3.set_title('Public Electricity'), ax3.set_ylabel('Count')
ax4.set_title('Chalkboard per Classroom')
plt.savefig('uni-categ-variables.png')

# Bivariate analysis of Management and Improved water supply
table = pd.crosstab(df['Management'], df['Improved_water_supply'])
table.plot(kind='bar', stacked=True,figsize=(8,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-mgt-water-supply.png')

# Bivariate analysis of Management and Improved sanitation
table = pd.crosstab(df['Management'], df['Improved_sanitation'])
table.plot(kind='bar', stacked=True,figsize=(8,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-mgt-sanitation.png')

# Bivariate analysis of Management and Electricity
table = pd.crosstab(df['Management'], df['Public_electricity'])
table.plot(kind='bar', stacked=True,figsize=(8,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-mgt-electricity.png')

# Bivariate analysis of Management and Chalkboard
table = pd.crosstab(df['Management'], df['Chalkboard/classroom'])
table.plot(kind='bar', stacked=True,figsize=(8,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-mgt-chalkboard-png')

# Bivariate analysis of State and Improved water supply
table = pd.crosstab(df['State'], df['Improved_water_supply'])
table.plot(kind='bar', stacked=True,figsize=(13,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-state-water-supply.png')

# Bivariate analysis of State and Improved sanitation
table = pd.crosstab(df['State'], df['Improved_sanitation'])
table.plot(kind='bar', stacked=True,figsize=(13,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-state-sanitation.png')

# Bivariate analysis of State and Electricity
table = pd.crosstab(df['State'], df['Public_electricity'])
table.plot(kind='bar', stacked=True,figsize=(13,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-state-electricity.png')

# Bivariate analysis of State and Chalkboard
table = pd.crosstab(df['State'], df['Chalkboard/classroom'])
table.plot(kind='bar', stacked=True,figsize=(13,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-state-chalkboard-png')

# Categorical variables comparison by Management
grouped = df.groupby('Management')[['Classrooms','Toilets','NCE_teachers','Fulltime_teachers']].sum().transpose()
grouped.plot(kind='bar', stacked=True, figsize=(13,6));
plt.ylabel('Frequency Distribution')
plt.savefig('biv-categ-mgt.png')

# Bivariate analysis of Male and Female Teachers
teachers = df[['Male_teachers', 'Female_teachers']].sum()
teachers.plot(kind='barh', color=['g', 'b']);
plt.title('Distribution of Teachers by Gender')
plt.xlabel('Frequency Distribution')
plt.savefig('biv-teachers-gender.png')

df_num = df[['Classrooms','Toilets',
              'NCE_teachers','Fulltime_teachers',
              'Male_teachers','Female_teachers','Total_students']]

# Heatmap of numerical variables Correlation
num_corr = df_num.corr()
#mask = num_corr < 0.5
plt.figure(figsize=(11,7))
sns.heatmap(num_corr, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, cmap='YlGnBu', annot_kws={"size": 8}, square=True);
plt.savefig('heatmap.png')

# Possible variables for further relationship analysis
df_num = df[['Classrooms','Toilets',
              'NCE_teachers','Fulltime_teachers',
              'Male_teachers','Female_teachers','Total_students']]
corr = df_num.corr()['Total_students'][:-1]
corr_fairly = corr[abs(corr) > 0.4]
print('There are {} fairly correlated values with Total Students:\n{}'.format(len(corr_fairly), corr_fairly))