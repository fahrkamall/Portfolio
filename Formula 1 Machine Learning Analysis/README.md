# METCS777TermProject

# Formula 1 Data Analysis Project - CS 777

## Overview
This project analyzes Formula One racing data through a comprehensive pipeline built on Azure Storage and Databricks. The data flow through structured layers: raw ingestion, transformation, analysis, and machine learning.

## Prerequisites
- An active Azure subscription.
- An active Databricks subscription.

## Azure Environment Setup
Before working with Databricks, you need to set up your Azure environment correctly:

1. Azure Active Directory (AAD) Setup:
   - Register an application in AAD to represent your Databricks application.
   - Obtain the Application (client) ID, Directory (tenant) ID, and Client Secret for your registered application.

2. Azure Storage Account:
   - Create an Azure Storage Account if you haven’t already done so.
   - Generate a Storage Account access key, which will be used to set up the mount points in Databricks.

3. Azure Databricks Workspace:
   - Set up an Azure Databricks Workspace.
   - Inside Databricks, create a secret scope which can securely store your Storage Account access key.

4. Mount Points:
   - Use the above credentials to create mount points in Databricks. This allows your Databricks workspace to access Azure Storage.

5. Permissions:
   - Ensure that the registered AAD application has the appropriate role assignments on the storage account to allow for reading and writing data.

## Repository Structure
- `Include`: Contains shared configurations and common functions.
- `Ingestion`: Scripts for ingesting data into the bronze layer.
- `ML Models`: Notebooks for machine learning models.
- `analysis`: Notebooks for data analysis and generating visualizations.
- `set-up`: Notebooks for setting up the environment, including mounting Azure Storage.
- `trans`: Notebooks for transforming data to the silver and gold layers.

## Getting Started
Initial Setup
- Start by executing notebooks in the `set-up` folder. These scripts will establish your data environment, including mounting Azure Data Lake Storage to Databricks.

Data Processing Workflow
- Follow the folder structure sequentially:
  - `Ingestion`: Load raw data.
  - `trans`: Process and transform the data.
  - `analysis`: Analyze the data and produce insights.
  - `ML Models`: Apply machine learning for predictions.

Running Notebooks
- Attach and run each notebook to a Databricks cluster.
- Follow instructions within each notebook, executing cells in order.

Notes
- Ensure that you replace any placeholder keys or credentials with your specific Azure details.
