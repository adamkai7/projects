from bs4 import BeautifulSoup
import csv
import os
import datetime

'''
Created by adam kainikara
The purpose of this analysis is to analyze passenger travel volumes from the TSA
(passenger volumes in the USA) to assess economic sentiment and indicators. If more people are
traveling now than in recent years, then ordinary people may be less worried about the economy
and have more disposable income.

Data extraction was performed using web scraping techniques implemented in Python, utilizing
the Beautiful Soup library to parse HTML tables containing passenger volume information. The
extraction script (extract_tsa.py) was designed to collect all available data. To ensure reliable
data retrieval and avoid being blocked by the website’s security measures, the script employed a
standard user agent string mimicking a legitimate browser request.


this is for if you want one mega file of all the data

files = ['passenger-volumes', '2024', '2023', '2022', '2021', '2020', '2019']
rows = []






to get the most recent year (rn 2025) do
wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"     -O ~/tmp/2025 https://www.tsa.gov/travel/passenger-volumes

all the other years are saved but if not do
for year in {2019..2024}; do
  wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36" \
       -O ~/tmp/$year https://www.tsa.gov/travel/passenger-volumes/$year
done

'''




for name in files:
    path = os.path.expanduser(f'~/tmp/{name}')
    
    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    table = soup.find('table')
    for tr in table.find_all('tr')[1:]:
        td = tr.find_all('td')
        if len(td) >= 2:
            date_str = td[0].text.strip()
            vol = td[1].text.strip().replace(',', '')
            rows.append([date_str, vol])

# sorts the rows by date ascending
rows.sort(key=lambda x: datetime.datetime.strptime(x[0], '%m/%d/%Y'))

with open('tsa.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Date', 'Passengers'])
    writer.writerows(rows)
