import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime

# Import Gas Prices data and sort on ascending date.
gas_prices = pd.read_csv("Weekly_U.S._All_Grades_All_Formulations_Retail_Gasoline_Prices.csv", encoding = "ISO-8859-1", skiprows = 4)
gas_prices["Date"] = pd.to_datetime(gas_prices["Date"])
gas_prices = gas_prices.sort_values(by = "Date", ascending = True)

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Plot all gas prices.
plt.rcParams['figure.dpi'] = 360
sns.lineplot(data=gas_prices, x="Date", y="All_Price", color = "k")

# Plot elections/political party change.
split_points = [datetime.date(1993,4,5), datetime.date(1996,9,5), datetime.date(2000,9,7), datetime.date(2004,9,2), datetime.date(2008,9,4), datetime.date(2012,9,6), datetime.date(2016,9,8), datetime.date(2020,9,3), datetime.date(2024,7,8)]
split_colors = ['blue', 'blue', 'red', 'red', 'blue', 'blue', 'red', 'blue'] 

dates = []
prices = []

# Finds closest future available data after election day. 
for date in split_points:
    temp_date = date
    while gas_prices[gas_prices["Date"] == str(temp_date)].size == 0:
        temp_date+=datetime.timedelta(days=1)
    dates.append(temp_date)
    prices.append(gas_prices[gas_prices["Date"] == str(temp_date)]["All_Price"].values[0])

# Plots average gas price over 4 year presidential election cycle.
election_day_df = pd.DataFrame({"Date" : dates, "All_Price" : prices})
sns.lineplot(data = election_day_df, x = "Date", y = "All_Price", color = "y")


# Plots background colors corresponding to party of executive branch. 
for i in range(len(split_points) - 1):
    plt.axvspan(split_points[i], split_points[i+1], facecolor=split_colors[i], alpha=0.2, zorder=-10,)

# Adds labels, legend, title to plot of all gas prices and presidential party.
plt.xlabel("")
plt.ylabel("Price (USD)")
plt.title("Price per Gallon of All Gasoline Grades Over Time")
sns.despine(left = True)
price_patch = mpatches.Patch(color='k', label='Price')
avg_patch = mpatches.Patch(color='y', label='Average Price')
red_patch = mpatches.Patch(color='r', label='Republican Presidency', alpha = 0.2,)
blue_patch = mpatches.Patch(color='b', label='Democratic Presidency', alpha = 0.2)
plt.legend(handles=[price_patch, avg_patch, red_patch, blue_patch], frameon = False)
plt.savefig('Price-Gallon All')
plt.clf()

# Plots historic mean price of gas throughout each month.
gas_prices["Month"] = pd.DatetimeIndex(gas_prices['Date']).month
plt.rcParams['figure.dpi'] = 360
pal = sns.color_palette("flare", 12)
data = gas_prices.groupby("Month")["All_Price"].mean()
rank = data.argsort().argsort()
fig, ax = plt.subplots(figsize = (8, 4))
sns.barplot(data = gas_prices, x = "Month", y = "All_Price", palette=(np.array(pal)[rank]).tolist(), ci = False)
plt.xlabel("")
ax.xaxis.set_ticks(np.arange(12)) # Simpy here to avoid UserWarning
ax.set_xticklabels(np.array(months).flatten())
ax.xaxis.set_ticks_position('none') 
plt.ylabel("Average Price (USD)")
plt.title("Average Price per Gallon of All Gasoline Grades per Month")
sns.despine(left = True)
plt.savefig('Price-Gallon Months')
plt.clf()


# Plots historic weekly average of gas price for each month.
gas_prices["Week"] = pd.cut(pd.DatetimeIndex(gas_prices['Date']).day, bins = [0, 7, 14, 21, 31], include_lowest = True, labels = ["1", "2", "3", "4"])
plt.rcParams['figure.dpi'] = 360
sns.set_style("darkgrid")
g = sns.FacetGrid(gas_prices, col = "Month", col_wrap=4)
g.map(sns.pointplot, "Week", "All_Price", order = [1,2,3,4], errorbar = None)
g.set_axis_labels(x_var = "Week", y_var = "Average Price")
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Average Price per Gallon of All Gasoline Grades per Week")
for idx,ax in enumerate(g.axes.flat):
        ax.set_title(months[idx])
plt.savefig('Price-Gallon Week')
plt.clf()

# Box plots of historic monthly gas prices. Dispersion indicates if "wild" changes in price occur.
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = gas_prices["Month"], y = gas_prices["All_Price"])
plt.xlabel("")
ax.xaxis.set_ticks(np.arange(12)) # Simpy here to avoid UserWarning
ax.set_xticklabels(np.array(months).flatten())
ax.xaxis.set_ticks_position('none') 
plt.ylabel("Price Range (USD)")
plt.title("Range of Price per Gallon of All Gasoline Grades per Month")
plt.savefig('Price-Gallon Range')
plt.clf()

# Determines which month has had the week with the highest gas price of the year historically. 
gas_prices["Year"] = pd.DatetimeIndex(gas_prices['Date']).year
indeces_max = gas_prices.groupby("Year")["All_Price"].idxmax()
max_prices = gas_prices.loc[indeces_max][["Month", "Week"]]
months_max = max_prices["Month"].value_counts()
print(f"How many times has a certain month had the most expensive week?\n{months_max}")
may_week = max_prices[max_prices["Month"] == 5]["Week"].value_counts()
print(f"What was the most expensive week of that year?\n{may_week}")

# Determines which month has had the week with the lowest gas price of the year historically. 
indeces_min = gas_prices.groupby("Year")["All_Price"].idxmin()
min_prices = gas_prices.loc[indeces_min][["Month", "Week"]]
months_min = min_prices["Month"].value_counts()
print(f"How many times has a certain month had the least expensive week?\n{months_min}")
jan_week = min_prices[min_prices["Month"] == 1]["Week"].value_counts()
dec_week = min_prices[min_prices["Month"] == 12]["Week"].value_counts()
print(f"What was the least expensive week of that year?\n{jan_week}")
print(f"What was the least expensive week of that year?\n{dec_week}")

# Plots linear regression on data throughout history.
gas_prices["ticks"] = range(0, len(gas_prices.index.values))
model = LinearRegression().fit(gas_prices[["ticks"]], gas_prices[["All_Price"]])
predictions = model.predict(gas_prices[["ticks"]])
predictions = pd.DataFrame(data = np.flip(predictions), columns = ["Pred_Price"])
gas_prices = gas_prices.join(predictions, how = "inner")
sns.lineplot(data = gas_prices, x = "Date", y = "Pred_Price", color = 'y')
sns.lineplot(data = gas_prices, x = "Date", y = "All_Price", color = 'k')
plt.xlabel("")
plt.ylabel("Price (USD)")
plt.title("Prediction of Price per Gallon of All Gasoline Grades")
pred = mpatches.Patch(color='y', label='Predicted Price')
actual = mpatches.Patch(color='k', label='Actual Price')
plt.legend(handles=[actual, pred])
plt.savefig('Price-Gallon Pred')
plt.clf()


# Predicts the gas prices for the next instance of a given month.
month_dfs = gas_prices.groupby("Month")
predictions = []
r2s = []
for name, group in month_dfs:
    month_price = pd.DataFrame()
    month_price["Price"] = group.groupby("Year")["All_Price"].mean()
    month_price["Ticks"] = range(0, len(month_price.index.values))
    X = month_price["Ticks"].values.reshape(-1, 1)
    y = month_price["Price"].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    prediction_data = np.array(len(month_price.index.values)).reshape(-1,1)
    est_price = model.predict(prediction_data)[0][0]
    predictions.append(est_price)
    r2 = r2_score(y, model.predict(X))
    r2s.append(r2)
# Plot monthly predictions.
print(f"Average R2 Score: {np.mean(r2s)}")
predictions = np.array(predictions)
idx = np.array([7,8,9,10,11,0,1,2,3,4,5,6])
future_months = np.array(months)[idx]
predictions = predictions[idx]
plt.figure(figsize=(10,8))
plt.plot(future_months, predictions)
plt.xlabel("")
plt.ylabel("Average Price (USD)")
plt.title("Predicted Average Price per Gallon of All Gasoline Grades per Month")
plt.savefig("Predicted Average Price per Gallon of All Gasoline Grades per Month", bbox_inches='tight')
plt.clf()















