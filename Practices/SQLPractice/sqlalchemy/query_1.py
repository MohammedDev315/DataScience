from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///doctors.db")

all_tables = engine.table_names()
print(all_tables)

doctors_data = pd.read_sql('SELECT * FROM doctors;', engine)
rates_data = pd.read_sql('SELECT * FROM rates;', engine)
left_join_data = \
    pd.read_sql('SELECT d.Name , d.Day FROM doctors d  LEFT JOIN rates r ON d.Day = r.Day and r.Location = d.Location' , engine)


print(doctors_data.shape)
print(rates_data.values)
print(left_join_data)



