# Databricks Migration Record

## Objective

Move the Madrid urban workflow from local Python and GeoPandas into an inspectable Databricks lakehouse while preserving source lineage, quality findings, analytical nulls, and interpretable model outputs.

## Completed architecture

```text
Uploaded files and remote-origin snapshots
  -> Unity Catalog raw_sources volume
  -> Bronze Delta tables
  -> Silver Delta tables
  -> Gold feature and model-output tables
  -> MLflow and gold_migration_review
```

The development namespace is `workspace.urban_workflow`. The bundle-managed job runs notebooks 01 through 06 in dependency order and passed end to end on 12 October 2026.

## Completed milestones

| Milestone | Result |
|---|---|
| 1. Ingestion | Five staged sources validated for structure, completeness, provenance, and SHA-256 |
| 2. Bronze | Five source-shaped Delta tables with deterministic snapshot writes and lineage |
| 3. Silver | Typed histories, canonical districts, explicit conflicts, and repaired boundaries |
| 4. Gold | Grid and district analytical feature tables with visible coverage and lineage |
| 5. Modeling | KMeans and IsolationForest outputs plus MLflow artifacts |
| 6. Review | Six-task Lakeflow Job and 28 passing migration checks |

## Accepted record counts

| Output | Records |
|---|---:|
| Bronze EMVS housing | 22 |
| Bronze/Silver population | 1,071 |
| Bronze/Silver indicators | 116,930 |
| Bronze/Silver boundaries | 21 |
| Bronze grid / Gold grid features | 10,100 |
| Silver district reference / housing | 21 each |
| Gold district features | 21 |
| Gold grid typology | 10,100 |
| Gold cluster profiles | 3 |
| Gold district cluster mix | 21 |
| Gold district anomalies | 21, including 2 exploratory flags |
| Gold migration review | 28 passing checks |

## Key decisions

- Free Edition cannot resolve `datos.madrid.es`, so two CKAN responses and one TopoJSON boundary file are manually uploaded and validated without misrepresenting them as Databricks downloads.
- Bronze preserves source values and geometry as WKB.
- Silver retains all population and indicator history. Eight conflicting indicator identities remain explicitly flagged rather than being silently resolved.
- Three invalid district polygons are repaired in Silver and retained with repair evidence.
- The legacy combined grid is classified as a `derived_snapshot`; the migration does not claim to reconstruct it.
- Gold retains all grid cells. The 433 cells without source district labels remain explicit, and analytical nulls are not converted to zero.
- Model outputs are exploratory and include preparation, lineage, evaluation, and interpretation metadata.

## Deferred work

### Dashboard / UI — not implemented

The original project included a Dash interface. This migration intentionally ends with reviewed Gold tables and successful orchestration. A Dash UI, Databricks App, or other presentation layer can be added later using the Gold contracts.

### Grid reconstruction

Rebuilding the legacy combined-grid pipeline from its original land-use, height, transport, rent, housing, and boundary sources remains an optional future track.
