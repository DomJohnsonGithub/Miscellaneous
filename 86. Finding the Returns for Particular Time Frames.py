import pandas as pd

# you will need yoUR own higher frequency data
dataframe = "dataframe consisting of returns for 5 minute data"


returns_2hr = dataframe.returns.resample('2H').sum()

# create a list of 2-hour intervals to iterate over
intervals = [('00:00', '02:00'), ('02:00', '04:00'), ('04:00', '06:00'),
             ('06:00', '08:00'), ('08:00', '10:00'), ('10:00', '12:00'),
             ('12:00', '14:00'), ('14:00', '16:00'), ('16:00', '18:00'),
             ('18:00', '20:00'), ('20:00', '22:00'), ('22:00', '00:00')]

# calculate the overall returns for each 2-hour interval
for interval in intervals:
    returns_interval = returns_2hr.between_time(interval[0], interval[1]).sum()
    print(f"Overall returns for {interval[0]}-{interval[1]}: {returns_interval}")