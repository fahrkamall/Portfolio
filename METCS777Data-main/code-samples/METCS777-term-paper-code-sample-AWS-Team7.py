# Databricks notebook source

# Loading AWS credentials from a CSV file stored in Databricks FileStore
aws_keys_df = spark.read.format('csv')\
    .option('header', 'true')\
    .option('inferschema', 'true')\
    .load('/FileStore/rootkey.csv')

# Displaying the column names of the DataFrame
aws_keys_df.columns

# Extracting AWS Access Key and Secret Key from the DataFrame
ACCESS_KEY = aws_keys_df.select('Access key ID').take(1)[0]['Access key ID']
SECRET_KEY = aws_keys_df.select('Secret access key').take(1)[0]['Secret access key']

# Importing the urllib library to encode the SECRET_KEY
import urllib
ENCODED_SECRET_KEY = urllib.parse.quote(string=SECRET_KEY, safe="")

# Setting up AWS S3 bucket details and mount name for Databricks
AWS_S3_BUCKET = 'databricks-5'
MOUNT_NAME = '/mnt/mount_s4'
SOURCE_URL = "s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_S3_BUCKET)

# Mounting the AWS S3 bucket to Databricks filesystem
dbutils.fs.mount(SOURCE_URL, MOUNT_NAME)

# Listing files in the mounted directory
# MAGIC %fs ls '/mnt/mount_s4'

# Reading a CSV file from the mounted S3 bucket into a Spark DataFrame(Bronze Level)
df = spark.read.option("header", "true").option("inferSchema", "true").csv("/mnt/mount_s4/circuits.csv")

# Renaming columns in the DataFrame for clarity (Silver Level, Transformations)
df2 = df.withColumnRenamed("circuitid", "circuit_Id")\
    .withColumnRenamed("circuitRef", "circuit_ref")

# Selecting specific columns from the DataFrame
from pyspark.sql.functions import *
df3 = df2.select(col("circuit_Id"), col("circuit_ref"), col("name"), col('location'), col('country'))

# Writing the modified DataFrame to a new Parquet file in the mounted S3 bucket
df3.write.mode("overwrite").parquet("/mnt/mount_s4/New_circuits")

# Reading the Parquet file into a Spark DataFrame (Gold Level)
AWS_s3 = spark.read.parquet("/mnt/mount_s4/New_circuits")

# Displaying the content of the DataFrame
AWS_s3.display()
