from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def to_upper(s):
    if s is not None:
        return s.upper()


# Register the UDF
to_upper_udf = udf(to_upper, StringType())
