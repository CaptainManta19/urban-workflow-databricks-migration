# Urban Workflow Databricks Migration

Migration of a Madrid urban-analysis workflow to Databricks using Unity Catalog, Delta tables, MLflow, Lakeflow Jobs, and Declarative Automation Bundles.

## Status and scope

The migration is complete through Bronze, Silver, Gold, modeling, and review. The six-task Lakeflow Job passed end to end on 12 October 2026, including the final 28-check scorecard.

This repository does **not** implement a dashboard, Dash UI, or Databricks App. The original project's Dash interface was intentionally not migrated. A presentation layer built on the Gold tables is optional future work.

```text
Unity Catalog volume
  -> Bronze source-shaped tables
  -> Silver cleaned entities and histories
  -> Gold analytical features and ML outputs
  -> MLflow and migration review scorecard
```

## Main outputs

| Layer | Outputs |
|---|---|
| Bronze | Five source-shaped tables with ingestion lineage |
| Silver | District reference, housing, population, indicators, and repaired boundaries |
| Gold | Grid features, district features, KMeans typologies, cluster profiles, district cluster mix, and anomaly scores |
| Review | `gold_migration_review` with 28 acceptance checks |

The workflow preserves 10,100 grid cells, including 433 without source district assignments. Missing height, rent, and housing values remain visible rather than being silently replaced.

## Important constraints

- Databricks Free Edition cannot reach `datos.madrid.es`; three complete remote-origin snapshots are manually staged and validated.
- The combined grid GeoPackage is a governed `derived_snapshot`. Reconstructing its legacy source pipeline is a separate future task.
- KMeans clusters are exploratory typologies. IsolationForest flags are review signals, not diagnoses.
- Raw source files, credentials, local data, and Databricks state are excluded from Git.

## Run the workflow

The development target uses `workspace.urban_workflow` and the `raw_sources` volume. Stage the five manifest-declared sources before running.

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev urban_workflow_job
```

When several CLI profiles use the same workspace, add `--profile "<profile name>"` to each command. Credentials remain in the local Databricks CLI profile.

Each successful notebook returns a short task summary. Failures retain their normal Databricks error details.

## Repository layout

```text
config/                 Source manifest
notebooks/              Six ordered workflow notebooks
resources/              Bundle-managed Lakeflow Job
src/urban_workflow/     Ingestion, Bronze, Silver, Gold, modeling, and review logic
tests/                  Local unit tests
docs/migration_plan.md  Completed migration record and deferred work
databricks.yml          Bundle configuration
```

## Local validation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python3 -m unittest discover -s tests
```

See [docs/migration_plan.md](docs/migration_plan.md) for accepted counts, milestone results, and design decisions.
