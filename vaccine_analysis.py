import pandas as pd           # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt    #for data visualization
import seaborn as sns               #data visualization library built over matplotlib
from sklearn.linear_model import LinearRegression

#reading data
df = pd.read_csv("C:\\Users\\Raman\\OneDrive\\Desktop\\country_vaccinations.csv")
#print one data rows of data set
print(df.loc[1])
#grouping data
new_df = df.groupby(["country",'iso_code','vaccines'])['total_vaccinations',
                     'people_vaccinated','people_fully_vaccinated',
                    'daily_vaccinations','total_vaccinations_per_hundred',
                    'people_vaccinated_per_hundred',"people_fully_vaccinated_per_hundred"
,'daily_vaccinations_per_million'].max().reset_index()
#top 10 vaccines used against covid-19
top10 = new_df['vaccines'].value_counts().nlargest(10)
top10
data = dict(new_df['vaccines'].value_counts(normalize = True).nlargest(10)*100) 

# The number of total vaccinations according to top 10 countries
data = new_df[['country','total_vaccinations']].nlargest(10,'total_vaccinations')
sns.barplot('country','total_vaccinations',data=data)
plt.xlabel("country")
plt.xticks(rotation=45)
plt.ylabel("total_vaccinations in 100 millions")
plt.show()
# The number of daily vaccinations
df1 = new_df[['country','daily_vaccinations']].nlargest(10,'daily_vaccinations')
sns.barplot('country','daily_vaccinations',data=df1)
plt.xlabel("country")
plt.xticks(rotation=45)
plt.ylabel("daily_vaccinations in millions")
plt.show()

# which vaccines is used by which country
dict = {}
for i in df.vaccines.unique():
    dict[i] = [df["country"][j] for j in df[df["vaccines"]==i].index]
vaccines = {}
for key, value in dict.items():
    vaccines[key] = set(value)
for i,j in vaccines.items():
    print(f"{i}:\n{j}")
new_data = df.dropna(axis = 0, how ='any')
print("Old data frame length:", len(df), "\nNew data frame length:",
       len(new_data), "\nNumber of rows with at least 1 NA value: ",
       (len(df)-len(new_data)))
# printing the names of the country
print("countries name")
country_names = df["country"].unique()
print(country_names)
#printing the count of each vaccines in dataset
print(df.vaccines.value_counts())
#regression line to show according to daily vaccinations what will be the total no. of vaccinated people
df2 = df.dropna()
x = df2.iloc[:, 7].values.reshape(-1,1)
y = df2.iloc[:, 4].values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_pred = linear_regressor.predict(x)
plt.scatter(x,y)
plt.xlabel('daily_vaccinations')
plt.ylabel('total no. of vaccinations')
plt.plot(x,y_pred,color='red')
plt.show()
#people fully vaccinated vs country
df1 = new_df[['country','people_fully_vaccinated']].nlargest(10,'people_fully_vaccinated')
sns.barplot('country','people_fully_vaccinated',data=df1)
plt.xlabel("country")
plt.xticks(rotation=45)
plt.ylabel("fully_vaccinated_people in millions")
plt.show()

print(new_df[new_df.total_vaccinations == new_df.total_vaccinations.max()])

x = new_df.daily_vaccinations.max()
print("maximum no. of people vaccinated in one day = ",x)

y = new_df.daily_vaccinations.min()
print("minimum number of people vaccinated in one day= ",y)

df4 = new_df[['country','total_vaccinations']]
print(df4.head())
print("country with maximum vaccination")
print(df4[df4.total_vaccinations == df4.total_vaccinations.max()])
print("country with minimum number of vaccinations")
print(df4[df4.total_vaccinations == df4.total_vaccinations.min()])

df5 = new_df[['country','people_fully_vaccinated']]
print("country with maximum number of fully vaccinated people")
print(df5[df5.people_fully_vaccinated == df5.people_fully_vaccinated.max()])
print("country with minimum number of fully vaccinated people")
print(df5[df5.people_fully_vaccinated == df5.people_fully_vaccinated.min()])

res = df['people_fully_vaccinated'].sum()
print("total no. of fully vaccinated people =",res)

res1 = df['total_vaccinations'].sum()
print("total no. of vaccinated people =",res1)

#extracting data of India
df_new = df[df['country'] == 'India']
df6 = df_new[['date','total_vaccinations','people_fully_vaccinated','daily_vaccinations']]
print(df6.head())
df.plot(x ='date', y='total_vaccinations', kind = 'line')
plt.xlabel('dates')
plt.ylabel('vaccinations in million')
print("maximum no. of vaccinated people in india in one day")
print(df6[df6.daily_vaccinations == df6.daily_vaccinations.max()])

print("minimum no. of vaccinated people in india in one day")
print(df6[df6.daily_vaccinations == df6.daily_vaccinations.min()])

total = df6['people_fully_vaccinated'].sum()
print("total no. of fully vaccinated people in india=",total)

total1 = df6['total_vaccinations'].sum()
print("total no. of vaccinated people in india=",total1)


#regression line to show according to daily vaccinations what will be the total no. of vaccinated people in india
df7 = df_new.dropna()
x = df7.iloc[:, 7].values.reshape(-1,1)
y = df7.iloc[:, 4].values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_pred = linear_regressor.predict(x)
plt.scatter(x,y)
plt.xlabel('daily_vaccinations')
plt.ylabel('total no. of vaccinations')
plt.plot(x,y_pred,color='purple')
plt.show()

#printing data from 15th january 2021 to 25th january 2021
df6['date'] = pd.to_datetime(df_new['date'])
m = (df6['date'] > '15-01-2021') & (df6['date'] < '25-01-2021')
m1 = df6.loc[m]
print(m1.head())
#plotting graph for daily vaccinations data extracted from specific dates
m1.plot(kind='bar',x='date',y='daily_vaccinations',color='green')
plt.xticks(rotation=45)
plt.xlabel('dates->')
plt.ylabel('daily vaccinations')
plt.show()
#plotting graph for total vaccinations data extracted from specific dates
m1.plot(kind='bar',x='date',y='total_vaccinations',color='red')
plt.xticks(rotation=45)
plt.xlabel('dates->')
plt.ylabel('total_vaccinations')
plt.show()
