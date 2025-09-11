from numpy import *
import datetime
import matplotlib.pyplot as plt
import csv
from volta.cmdparser import CommandParser

## 

'''
created by Adam Kainikara
The analysis script (tsa_data_analysis.py) processes data by converting calendar dates to day
of-year indices (1-366) to facilitate year-over-year comparisons of seasonal patterns. This approach
enables direct comparison of specific calendar periods across different years, accounting for leap
years and ensuring consistent alignment.

EXAMPLE ON HOW TO RUN THE FILE

FIRST NEED TO DO THE extract_tsa.py part first to get the csv data if havent already.
then

python view_tsa_part2.py --days 10

where --days and 10 is the window size. it runs in half on either side of the day ie if the day was jul 25,
the window would be 5 days before jul 25 and 5 days after jul 25
'''
def read_tsa_file_data(year):
    # look at a single year's TSA passenger data from its CSV file
    passenger_counts_v = full(366, nan)
    
    with open(f'{year}.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header row
        
        for row in reader:
            date_obj_v = datetime.datetime.strptime(row[0], '%m/%d/%Y')
            day_index_v = date_obj_v.timetuple().tm_yday - 1
            passenger_counts_v[day_index_v] = int(row[1])
            
    return passenger_counts_v

def centered_rolling_mean(data_array_v, window_size_v):
   #calculates a centered rolling average for the data
    rolling_average_v = full(len(data_array_v), nan)
    half_window_v = window_size_v // 2
    
    for i_v in range(half_window_v, len(data_array_v) - half_window_v):
        window_v = data_array_v[i_v - half_window_v : i_v + half_window_v + 1]
        rolling_average_v[i_v] = mean(window_v)
        
    return rolling_average_v

def analyze_tsa_data(years_v, window_size_v):
    all_results_v = {}
    for year_v in years_v:
        daily_counts_v = read_tsa_file_data(year_v)
        all_results_v[year_v] = centered_rolling_mean(daily_counts_v, window_size_v)
    return all_results_v

## -------------------------------------------------------------------
## Plotting
## -------------------------------------------------------------------

def print_analysis(results_v, window_size_v):
    #prints results 
    years_v = sorted(results_v.keys())
    radius_v = window_size_v // 2
    
    for day_index_v in range(366):
        has_data_v = any(not isnan(results_v[year_v][day_index_v]) for year_v in years_v)
        if not has_data_v:
            continue

        date_obj_v = datetime.datetime.strptime(f'2024-{day_index_v + 1}', '%Y-%j')
        print(f"\nDay {date_obj_v.strftime('%b %d')} (±{radius_v} days):")
        
        previous_year_average_v = None
        previous_year_v = None
        
        for year_v in years_v:
            average_v = results_v[year_v][day_index_v]
            
            if isnan(average_v):
                print(f"  {year_v}: no data")
            else:
                message_v = f"  {year_v}: {average_v:,.0f} passengers"
                if previous_year_average_v is not None and previous_year_average_v > 0:
                    gap_v = average_v - previous_year_average_v
                    percent_change_v = (gap_v / previous_year_average_v) * 100
                    message_v += f" (gap: {gap_v:+,.0f}, {percent_change_v:+.1f}% from {previous_year_v})"
                
                print(message_v)
                previous_year_average_v = average_v
                previous_year_v = year_v

def plot_metric(results_v, metric_v='ratio', title_suffix_v='', window_size_v=None):
    #plot the raito andpercent graphs
    years_v = sorted(results_v.keys())
    days_v = arange(366)
    dates_v = [datetime.datetime.strptime(f'2024-{int(d) + 1}', '%Y-%j') for d in days_v]

    plt.figure(figsize=(12, 6))

    for i_v in range(1, len(years_v)):
        curr_year_v, prev_year_v = years_v[i_v], years_v[i_v-1]
        curr_data_v, prev_data_v = results_v[curr_year_v], results_v[prev_year_v]

        valid_v = (~isnan(curr_data_v)) & (~isnan(prev_data_v)) & (prev_data_v != 0)

        vals_v = full(366, nan)
        if metric_v == 'ratio':
            vals_v[valid_v] = curr_data_v[valid_v] / prev_data_v[valid_v]
            ylabel_v = "Ratio"
            hline_v = 1
        else:  # percent_change
            vals_v[valid_v] = (curr_data_v[valid_v] - prev_data_v[valid_v]) / prev_data_v[valid_v] * 100
            ylabel_v = "Percent Change (%)"
            hline_v = 0

        plt.plot(dates_v, vals_v, label=f"{curr_year_v} vs {prev_year_v}")

    plt.axhline(hline_v, color='gray', linestyle='--', linewidth=1)

    if window_size_v:
        plt.title(f"Passenger Count {ylabel_v} {title_suffix_v} (Window Size: {window_size_v} days)")
    else:
        plt.title(f"Passenger Count {ylabel_v} {title_suffix_v}")

    plt.xlabel("Month")
    plt.ylabel(ylabel_v)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.show()


def plot_all(results_with_2020_v, results_without_2020_v, window_size_v=None):
    plot_metric(results_with_2020_v, metric_v='ratio', title_suffix_v="(Including 2020)", window_size_v=window_size_v)
    plot_metric(results_with_2020_v, metric_v='percent_change', title_suffix_v="(Including 2020)", window_size_v=window_size_v)
    plot_metric(results_without_2020_v, metric_v='ratio', title_suffix_v="(Excluding 2020)", window_size_v=window_size_v)
    plot_metric(results_without_2020_v, metric_v='percent_change', title_suffix_v="(Excluding 2020)", window_size_v=window_size_v)


def parse_args_args(): 
    cmdparser_v = CommandParser()
    cmdparser_v.arg('--days', type=int, help='Window size in days')
    args_v = cmdparser_v.eval()
    return args_v


def main():
    args_v = parse_args_args()

    print("=== With 2020 included ===")
    years_with_2020_v = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
    results_with_2020_v = analyze_tsa_data(years_with_2020_v, args_v.days)
    print_analysis(results_with_2020_v, args_v.days)

    print("\n=== Without 2020 ===")
    years_without_2020_v = ['2019', '2021', '2022', '2023', '2024', '2025']
    results_without_2020_v = analyze_tsa_data(years_without_2020_v, args_v.days)
    print_analysis(results_without_2020_v, args_v.days)

    plot_all(results_with_2020_v, results_without_2020_v, window_size_v=args_v.days)


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

