import wrds

db = wrds.Connection(wrds_username='deepshah')

result = db.raw_sql("select * from cboe.cboe limit 10")
print(result)