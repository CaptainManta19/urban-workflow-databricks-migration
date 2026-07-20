# Urban Workflow Databricks Migration

This repository is the Databricks/lakehouse migration of the earlier `urban-workflow-implementation` project.

The goal is to move the local Madrid urban data workflow into a Databricks-friendly structure step by step, while preserving transparency, provenance, warnings, and human review checkpoints.

## Project Direction

This repo starts with a simple exploratory Databricks setup:

- GitHub repository linked to a Databricks Git folder
- Databricks notebooks for visible workflow steps
- Reusable Python modules under `src/urban_workflow/`
- Delta tables for Bronze, Silver, and Gold layers later
- No Databricks Asset Bundles at the beginning

Databricks Jobs, Databricks Asset Bundles, and Databricks Apps may be added later if the workflow needs them.

## Target Architecture

External/local Madrid sources  
-> Bronze Delta tables: raw collected data plus provenance  
-> Silver Delta tables: cleaned normalized entities  
-> Gold Delta tables: grid metrics, district metrics, and ML outputs  
-> dashboard or Databricks App later

## Repository Structure

```text
src/urban_workflow/
  ingestion/    Source collection and provenance logic
  features/     Grid-level and district-level feature engineering
  models/       Clustering and anomaly detection
  quality/      Validation, warnings, and review checks
  utils/        Shared helpers

notebooks/
  01_ingest_sources.ipynb
  02_build_bronze.ipynb
  03_build_silver.ipynb
  04_build_gold_features.ipynb
  05_run_ml.ipynb
  06_review_outputs.ipynb

docs/
  migration_plan.md

app/
  dash_app.py
  streamlit_app.py