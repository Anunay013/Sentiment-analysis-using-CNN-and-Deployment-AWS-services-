import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from tweet import TextPreprocessing

print('--IMPORTED--')

# Define text_pre_processor
processor = TextPreprocessing()

print('--PROCESSOR DEFINED--')

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "a6_db", table_name = "train", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "a6_db", table_name = "train", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]

print('--IMPORTED TABLE--')

applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")

print('--COLUMNS FILTERED--')

# Process tweets
def process_rows(dynamicRecord):
    dynamicRecord["features"] = processor.process_tweet(dynamicRecord["tweet"])
    return dynamicRecord
processed_tweets = Map.apply(frame = applymapping1, f = process_rows, transformation_ctx = "processed_tweets")

print('--TWEETS PROCESSED--')

processed_tweets_coalesced = processed_tweets.coalesce(1)

print('--COALESCED--')

## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://ai-ops-skand-a6/output_data/train"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = processed_tweets_coalesced, connection_type = "s3", connection_options = {"path": "s3://ai-ops-skand-a6/output_data/train"}, format = "json", transformation_ctx = "datasink2")

print('--SINKED--')

job.commit()