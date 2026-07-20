# Migration Plan

This document tracks the migration from the original local Madrid urban workflow to a Databricks/lakehouse-style project.

## Source Project

Original repository: `urban-workflow-implementation`

The original workflow includes:

- provenance-aware source collection
- human review checkpoints
- feature engineering at grid and district levels
- KMeans clustering for urban form/access typology
- IsolationForest anomaly detection for district-level socio-spatial mismatch
- dashboard inspection

## Migration Repository

Current repository: `urban-workflow-databricks-migration`

This repo is used as the Git-backed file base for Databricks.

## Guiding Principles

- Preserve provenance and warnings as first-class data.
- Keep the workflow transparent and auditable.
- Avoid overengineering early.
- Prefer notebooks for visible Databricks workflow steps.
- Put reusable logic in `src/urban_workflow/`.
- Replace local file outputs gradually with Delta tables.
- Keep dashboard migration out of the first phase.



## Databricks Target Architecture

```text
External/local Madrid sources
  -> Bronze Delta tables
      raw collected data
      provenance metadata
      collection warnings
  -> Silver Delta tables
      cleaned normalized entities
      harmonized schemas
  -> Gold Delta tables
      grid metrics
      district metrics
      ML outputs
  -> dashboard / Databricks App later
```

