import pandas as pd
import math
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


#importing data that we use for analysis

col_list = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']
data = pd.read_csv('flights.csv', usecols = col_list)
airlines = pd.read_csv('airlines.csv')

airline_delay = {}
half_hour_delay = {}
canceled = {}

total_flights = {}

#separately counting the number of flights, delays, number of 30+ minutes delay and the cancelation of flights in regard to a airline

for i in range(len(data['AIRLINE'])):
    if data['AIRLINE'][i] in total_flights.keys():
        total_flights[data['AIRLINE'][i]] += 1
    else:
        total_flights[data['AIRLINE'][i]] = 1
    if data['DEPARTURE_DELAY'][i] > 0:
        if data['AIRLINE'][i] in airline_delay.keys():
            airline_delay[data['AIRLINE'][i]] += 1
        else:
            airline_delay[data['AIRLINE'][i]] = 1
    
    if data['DEPARTURE_DELAY'][i] > 30:
        if data['AIRLINE'][i] in half_hour_delay.keys():
            half_hour_delay[data['AIRLINE'][i]] += 1
        else:
            half_hour_delay[data['AIRLINE'][i]] = 1
    
    if math.isnan(data['DEPARTURE_DELAY'][i]) == True:
        if data['AIRLINE'][i] in canceled.keys():
            canceled[data['AIRLINE'][i]] += 1
        else:
            canceled[data['AIRLINE'][i]] = 1

#counting probabilities

delay_prob = {}
half_hour_delay_prob = {}
canceled_prob = {}

for comp in total_flights.keys():
    delay_prob[comp] = airline_delay[comp]/total_flights[comp]
    half_hour_delay_prob[comp] = half_hour_delay[comp]/total_flights[comp]
    canceled_prob[comp] = canceled[comp]/total_flights[comp]

#sorting our dictionaries in descending order
airline_delay = sorted(airline_delay.items(), key = lambda x : x[1], reverse = True)
half_hour_delay= sorted(half_hour_delay.items(), key = lambda x : x[1], reverse = True)
canceled= sorted(canceled.items(), key = lambda x : x[1], reverse = True)


#matching first five companies to their actual names and ploting the overall results 
first_five = airline_delay[:5]
company = []
time = []
for i in range(5):
    company.append(first_five[i][0])
    time.append(first_five[i][1])
for i in range(len(company)):
    for j in range(len(airlines['IATA_CODE'])):
        if company[i] == airlines['IATA_CODE'][j]:
            company[i] = airlines['AIRLINE'][j]


plt.bar(company, time, width = 0.8, color = 'rgbky')
plt.xticks(rotation=45)
plt.title('Overall Delay')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Delay')

first_five_delay = half_hour_delay[:5]
company1 = []
time1 = []
for i in range(5):
    company1.append(first_five_delay[i][0])
    time1.append(first_five_delay[i][1])
for i in range(len(company1)):
    for j in range(len(airlines['IATA_CODE'])):
        if company1[i] == airlines['IATA_CODE'][j]:
            company1[i] = airlines['AIRLINE'][j]

plt.bar(company1, time1, width = 0.8, color = (0.1, 0.1, 0.1, 0.1),  edgecolor='red')
plt.xticks(rotation=45)
plt.title('Half hour delay')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Half hour delay')


first_five_canceled = canceled[:5]
company2 = []
time2 = []
for i in range(5):
    company2.append(canceled[i][0])
    time2.append(first_five_canceled[i][1])
for i in range(len(company2)):
    for j in range(len(airlines['IATA_CODE'])):
        if company2[i] == airlines['IATA_CODE'][j]:
            company2[i] = airlines['AIRLINE'][j]

plt.bar(company2, time2, width = 0.8, color = [np.random.rand(3,)])
plt.xticks(rotation=45)
plt.title('Number of cancelations')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Number of cancelations')

#doing the same thing with probabilities

delay_prob = sorted(delay_prob.items(), key = lambda x : x[1], reverse = True)
half_hour_delay_prob= sorted(half_hour_delay_prob.items(), key = lambda x : x[1], reverse = True)
canceled_prob= sorted(canceled_prob.items(), key = lambda x : x[1], reverse = True)

ldp = delay_prob[-1]
lhdp = half_hour_delay_prob[-1]
lcp = canceled_prob[-1]

for i in range(len(airlines['IATA_CODE'])):
    if airlines['IATA_CODE'][i] == ldp[0]:
        print('Company with lowest overall delay probability is {} with probability {}'.format(airlines['AIRLINE'][i],ldp[1]))
    if airlines['IATA_CODE'][i] == lhdp[0]:
        print('Company with lowest probability of delay of more than half hour is {} with probability {}'.format(airlines['AIRLINE'][i],lhdp[1]))
    if airlines['IATA_CODE'][i] == lcp[0]:
        print('Company with lowest probability of cancelation is {} with probability {}'.format(airlines['AIRLINE'][i],lcp[1]))
        
first_five_p = delay_prob[:5]
company_p = []
time_p = []
for i in range(5):
    company_p.append(first_five_p[i][0])
    time_p.append(first_five_p[i][1])
for i in range(len(company_p)):
    for j in range(len(airlines['IATA_CODE'])):
        if company_p[i] == airlines['IATA_CODE'][j]:
            company_p[i] = airlines['AIRLINE'][j]

plt.bar(company_p, time_p, width = 0.8, color = 'rgbky')
plt.xticks(rotation=45)
plt.title('Overall Delay Probability')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Delay Prob')
plt.close()

first_five_delay_p = half_hour_delay_prob[:5]
company1_p = []
time1_p = []
for i in range(5):
    company1_p.append(first_five_delay_p[i][0])
    time1_p.append(first_five_delay_p[i][1])
for i in range(len(company1_p)):
    for j in range(len(airlines['IATA_CODE'])):
        if company1_p[i] == airlines['IATA_CODE'][j]:
            company1_p[i] = airlines['AIRLINE'][j]

plt.bar(company1_p, time1_p, width = 0.8, color = (0.1, 0.1, 0.1, 0.1),  edgecolor='red')
plt.xticks(rotation=45)
plt.title('Half hour delay probability')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Half hour delay probability')
plt.close()

first_five_canceled_p = canceled_prob[:5]
company2_p = []
time2_p = []
for i in range(5):
    company2_p.append(first_five_canceled_p[i][0])
    time2_p.append(first_five_canceled_p[i][1])
for i in range(len(company2_p)):
    for j in range(len(airlines['IATA_CODE'])):
        if company2_p[i] == airlines['IATA_CODE'][j]:
            company2_p[i] = airlines['AIRLINE'][j]

plt.bar(company2_p, time2_p, width = 0.8, color = [np.random.rand(3,)])
plt.xticks(rotation=45)
plt.title('Probability of cancelations')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Probability of cancelations')
plt.close()



#importing other part of data that we use for analysis
col_list = ['MONTH','DAY_OF_WEEK', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']
data = pd.read_csv('flights.csv', usecols = col_list)

#ploting the mean values

grouped = data.groupby('MONTH').mean()
months = ['january', 'february','march','april','may','june','july','august','september','october','november','december']

plt.plot(months, grouped['DEPARTURE_DELAY'], 'bo')
plt.plot(months, grouped['ARRIVAL_DELAY'],'rx')
plt.xticks(rotation=45)
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.title('Departure and Arrival Delay')
plt.savefig('monthly')

count_month = data.groupby('MONTH').size()

numbers = []
for i in range(len(data['MONTH'])):
    if data['DEPARTURE_DELAY'][i] <= 0:
        numbers.append(i)
data = data.drop(numbers)
data = data.dropna()

count_month_delay = data.groupby('MONTH').size()

#ploting numbers of delay per month
figure = sns.countplot(x='MONTH', data=data)
plt.title('Number of Delays per Month')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('count')
plt.show()

probability = []
for i in range(12):
    probability.append(count_month_delay[i+1]/count_month[i+1])
    
plt.bar(months,probability)    
plt.xticks(rotation=45)
plt.title('Probability of delay per month')
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Probability of delay per month')
plt.close()
   
#dividing dataset in two parts, weekend and work days  

weekend = pd.concat([data [data.DAY_OF_WEEK == 6],data [data.DAY_OF_WEEK == 7]])
work_day = pd.concat([data [data.DAY_OF_WEEK == 1],data [data.DAY_OF_WEEK == 2], data [data.DAY_OF_WEEK == 3], data [data.DAY_OF_WEEK == 4], data [data.DAY_OF_WEEK == 5]])

weekend = weekend.iloc[:,1:]
work_day = work_day.iloc[:,1:]

weekend_delay = np.where(np.isnan(weekend['DEPARTURE_DELAY']) == True, 'Canceled', 
                    np.where(weekend['DEPARTURE_DELAY'] == 0, 'On time',
                    np.where(weekend['DEPARTURE_DELAY'] < 0, 'Left early', 
                             np.where(weekend['DEPARTURE_DELAY'] > 30, 'Significant delay', 'Slight delay'))))

work_day_delay = np.where(np.isnan(work_day['DEPARTURE_DELAY']) == True, 'Canceled', 
                    np.where(work_day['DEPARTURE_DELAY'] == 0, 'On time',
                    np.where(work_day['DEPARTURE_DELAY'] < 0, 'Left early',
                             np.where(work_day['DEPARTURE_DELAY'] > 30, 'Significant delay', 'Slight delay'))))



weekend_days = pd.DataFrame(weekend_delay)
work_day_days = pd.DataFrame(work_day_delay)
weekend_days = weekend_days.rename(columns = {0:'Departure'})
work_day_days = work_day_days.rename(columns = {0:'Departure'})

weekend_percentage = weekend_days.Departure.value_counts()/len(weekend_days.Departure)*100
work_day_percentage = work_day_days.Departure.value_counts()/len(work_day_days.Departure)*100

weekend_percentage.plot.bar(color = 'y')
plt.title('Percentage of departure on weekends')
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel('Percentage [%]')
plt.tight_layout()
plt.savefig('weekend%')

work_day_percentage.plot.bar(color = 'g')
plt.title('Percentage of departure on work days')
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel('Percentage [%]')
plt.tight_layout()
plt.savefig('work_day%')






























