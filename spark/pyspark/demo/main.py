#!/usr/bin/env python

from pyspark.sql import SparkSession
from udf import to_upper_udf


def main():
    spark = SparkSession.builder.appName("PySpark Demo").getOrCreate()

    df = spark.createDataFrame([
        {"name": "Michael"},
        {"name": "Andy", "age": 30},
        {"name": "Justin", "age": 19}
    ])

    df2 = df.withColumn("name_upper", to_upper_udf(df["name"]))

    df2.show()


if __name__ == '__main__':
    main()
