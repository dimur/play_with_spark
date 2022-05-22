from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import calendar
from datetime import datetime
from datetime import timedelta

spark = SparkSession.builder.appName("SparkSQLHomework").master("local[2]").getOrCreate()

df = (
    spark.read
        .option('header', True)
        .option('sep', ',')
        .option('inferSchema', True)
        .csv('owid-covid-data.csv')
)

# 1. Выберите 15 стран с наибольшим процентом переболевших на 31 марта
# (в выходящем датасете необходимы колонки: iso_code, страна, процент переболевших)
print("Task 1\n")
report_date = '2021-03-31'
top_15 = 15

df1 = df.select('iso_code', 'location', 'date', 'total_cases', 'population') \
    .where(F.length('iso_code') <= 3) \
    .where(F.col('date') == report_date) \
    .withColumn('percentage of recoveries', F.round((F.col('total_cases') / F.col('population')) * 100, 2)) \
    .select('iso_code', 'location', 'percentage of recoveries') \
    .sort(F.col('percentage of recoveries').desc())

df1.show(top_15)


# 2. Top 10 стран с максимальным зафиксированным кол-вом новых случаев за последнюю неделю марта 2021 в отсортированном порядке по убыванию
# (в выходящем датасете необходимы колонки: число, страна, кол-во новых случаев)

# Функция принимает кортеж (%Y, %m)
# и возвращает список дат последней недели месяца строками формата '%Y-%m-%d'
print("Task 2\n")
def get_last_week_dates(target_month):
    target_month_last_day = calendar.monthrange(*target_month)[1]
    last_day_weekday = calendar.weekday(*target_month, target_month_last_day)
    result = list(range(last_day_weekday + 1))
    result = [f"{target_month[0]}-{target_month[1]:02}-{target_month_last_day - x}" for x in result]
    return result


target_month = (2021, 3)
top_10 = 10
last_week_dates = get_last_week_dates(target_month)

windowLocation = Window.partitionBy('location').orderBy(F.col('new_cases').desc())
df2 = df.select('date', 'location', 'new_cases') \
    .where(F.length('iso_code') <= 3) \
    .filter(df.date.isin(last_week_dates)) \
    .withColumn('row', F.row_number().over(windowLocation)) \
    .filter(F.col('row') == 1).drop('row') \
    .sort(F.col('new_cases').desc())

df2.show(top_10)


# 3. Посчитайте изменение случаев относительно предыдущего дня в России за последнюю неделю марта 2021.
# (например: в россии вчера было 9150 , сегодня 8763, итог: -387)
# (в выходящем датасете необходимы колонки: число, кол-во новых случаев вчера, кол-во новых случаев сегодня, дельта)
print("Task 3\n")

# Функция принимает строку с датой в формате '%Y-%m-%d'
# и возвращает строку с предыдущей датой в формате '%Y-%m-%d'
def previous_date(current_date):
    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=1)
    return current_date_obj.strftime('%Y-%m-%d')


target_month = (2021, 3)
last_week_dates = get_last_week_dates(target_month)

window = Window.partitionBy().orderBy(F.col('date'))
df3 = df.select('date', 'new_cases') \
    .where(F.col('iso_code') == 'RUS') \
    .filter(df.date.isin(last_week_dates + [previous_date(last_week_dates[-1])])) \
    .withColumn("prev_date_new_cases", F.lag(df.new_cases).over(window))

df3 = df3.withColumn("delta", F.when(F.isnull(df3.new_cases - df3.prev_date_new_cases), 0).otherwise(
    df3.new_cases - df3.prev_date_new_cases)) \
    .filter(df.date.isin(last_week_dates)) \
    .select('date', 'prev_date_new_cases', 'new_cases', 'delta')

df3.show()

spark.stop()
