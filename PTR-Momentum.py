import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors

# HYPERPARAMETERS
ticker = "NG=F"
history = 10 #years
#cutoff percentage
zigzag_p = 0.5
#regulates the smoothness
gaussian_smoother = 12
#insert the current close into the the zigzag_df, use False when backtesting
insert_today = False 


# DOWNLOAD HISTORICAL DATA
# Assuming 'today' is a Pandas Timestamp
today = pd.to_datetime("today")
# Converting Timestamp to string
today_str = today.strftime('%Y-%m-%d')
# Define the end date as a datetime object; YOU CAN CHANGE DATE HERE FOR BACKTESTING, i.e. "2022-08-20" instead of today_str
end_date = datetime.strptime(today_str, "%Y-%m-%d")
 #datetime.today() - timedelta(days=)
start_date = end_date - timedelta(days=history*365)

# Format the dates as strings
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")   

# Download the Weekly data
df = yf.download(ticker, start=start_date_str, end=end_date_str, interval='1d')
df.index = pd.to_datetime(df.index)


# ZIGZAG CODE
#cutoff multiplier for zigzag calculation
zph = (100 + zigzag_p) / 100
zpl = (100 - zigzag_p) / 100

# Defining a temporary high and a dictionary for temporary highs
tmp_high = df["Adj Close"].iloc[0]
tmp_highs_d = {}

# Defining a temporary low and a dictionary for temporary lows
tmp_low = df["Adj Close"].iloc[0]
tmp_lows_d = {}

# Defining the zigzag dataframe for concatination of highs and lows
zigzag_df = pd.DataFrame({"date":df.index[0], "close": [df["Adj Close"].iloc[0]]})

# Iteration trough each row of the yfinance dataframe with price data 
for index, row in df.iterrows():
    # Defining date and close
    date = index
    date_str = date.strftime('%Y-%m-%d')
    close = row["Adj Close"]
    
    # if condition for filling the temporary high dictionary
    if close > tmp_high and close > tmp_low * zph:
            # condition is fulfilled close is now tmp_high
            tmp_high = close
            # new tmp_low threshold for collection of temporary lows
            tmp_low = close * zpl
            # adding new key value pair into tmp_highs_d
            tmp_highs_d[date_str] = tmp_high
            print("highs_d: ", tmp_highs_d)
            
            # make sure tmp_lows_d is not empty
            if not tmp_lows_d:
                print("lows_d is empty")
            # get the lowest value with date from the tmp_lows_d, turn it into a dataframe
            # concat the tl_df df to the zigzag_df
            else:
                lowest_key = min(tmp_lows_d, key=tmp_lows_d.get)
                lowest_value = tmp_lows_d[lowest_key]
                tl_df = pd.DataFrame({"date": [lowest_key], "close": [lowest_value]})
                zigzag_df = pd.concat([zigzag_df, tl_df])
                tmp_lows_d = {}
            
            

    if close < tmp_low and close < tmp_high * zpl: 
            tmp_low = close
            tmp_high = close * zph

            tmp_lows_d[date_str] = tmp_low
            
            
            if not tmp_highs_d:
                print("highs_d is empty")
            else:
                highest_key = max(tmp_highs_d, key=tmp_highs_d.get)
                highest_value = tmp_highs_d[highest_key]
                th_df = pd.DataFrame({"date": [highest_key], "close": [highest_value]})
                zigzag_df = pd.concat([zigzag_df, th_df])
                tmp_highs_d = {}


# PTR CODE
# Inserts the current close as last row in the zigzag_df if insert_today = True
if insert_today == True: 
    last_row_in_df = df.iloc[-1]
    current_close = last_row_in_df['Adj Close']
    new_row_for_zz = pd.DataFrame({
        "date": [today_str],
        "close": [current_close]
        })
    
    # Append new_row to zigzag_df
    zigzag_df = pd.concat([zigzag_df, new_row_for_zz], ignore_index=True)

# Create a separate 'dates' column before setting 'date' as index to calculate datetime differences
zigzag_df["date"] = pd.to_datetime(zigzag_df["date"])
zigzag_df["dates"] = zigzag_df["date"]  # Separate 'dates' column

# Set 'date' as index
zigzag_df = zigzag_df.set_index("date")

# Calculate time differences using the 'dates' column
zigzag_df['time_diff'] = zigzag_df["dates"].diff()
zigzag_df["time_diff"] = zigzag_df["time_diff"].dt.total_seconds()
zigzag_df["price_diff"] = zigzag_df["close"].diff()
zigzag_df["price_time_ratio"] = zigzag_df["price_diff"] / zigzag_df['time_diff']
zigzag_df["price_time_ratio"] = zigzag_df["price_time_ratio"].ffill()
zigzag_df["price_time_ratio"] = gaussian_filter1d(zigzag_df['price_time_ratio'], sigma=gaussian_smoother)

# Normalize ptr values to a range between 0 and 1 for the background colors on the first plot
norm = mcolors.Normalize(vmin=zigzag_df['price_time_ratio'].min(), vmax=zigzag_df['price_time_ratio'].max())
# Create a colormap that transitions from red to green (RdYlGn)
cmap = plt.cm.RdYlGn

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)  # 2 rows, 1 column

# First subplot: Matplotlib line plot for 'Adj Close' with Value of PTR momentum as background positive green, negativ in red
# Adding vertical lines for each day based on ptr values
for i, row in zigzag_df.iterrows():
    # Get the color based on ptr value
    color = cmap(norm(row['price_time_ratio']))
    # Plot a vertical line on the given date
    axs[0].axvline(row.name, color=color, linewidth=5, alpha=0.05)
# Plot Adj Close in front of the background colors
axs[0].plot(df.index, df["Adj Close"], label='Adj Close', color='black')
# Customize the first subplot
axs[0].set_title(f'Adjusted Close with PTR-Momentum Background Color ({ticker})')
axs[0].set_ylabel('Adjusted Close')
axs[0].legend()
axs[0].grid(True)

# Second subplot: Seaborn line plot
sns.lineplot(x=zigzag_df.index, y=zigzag_df["price_time_ratio"], ax=axs[1], label='PTR', color='blue')
axs[1].set_title('Line Plot of PTR-Momentum Indicator')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('ptr')
# Fill the area between 0 and ptr where ptr is positive
axs[1].fill_between(zigzag_df.index, 0, zigzag_df["price_time_ratio"], where=(zigzag_df["price_time_ratio"] > 0), color='green', alpha=0.3)
# Fill the area between 0 and ptr where ptr is negative
axs[1].fill_between(zigzag_df.index, 0, zigzag_df["price_time_ratio"], where=(zigzag_df["price_time_ratio"] < 0), color='red', alpha=0.3)
# Show the plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
