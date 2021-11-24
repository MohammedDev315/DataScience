from bs4 import BeautifulSoup
import requests
import csv

# csv_file = open("result2.csv" , "w")
# write_to_csv = csv.writer(csv_file)
# write_to_csv.writerow(['image_link' , 'title' , 'text' , 'post_time' ])

url_1 = "https://www.flyin.com/flight/result?tripType=1&dep=JED&arr=AHB&adt=1&chd=0&inf=0&cbn=e&depdate=2021-11-28&airlineType=&isDirectFlights=true&cityId=51173&flexDates=false"

req1 = requests.get(url_1).text
soup2 = BeautifulSoup(req1, "html5lib")
result1 = soup2.find("div")
print(result1)

# csv_file.close()



