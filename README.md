# MDB Waterbird Data Pipeline — X+GeoAI Project

Reproducible Python pipeline for harmonising 4 waterbird survey datasets
into a single modelling-ready master table, with site-specific subsets
for Macquarie Marshes and Gwydir Wetlands (Ramsar sites, NSW, Australia).

## Datasets (place in data/raw/)

| File (rename to) | Source |
|---|---|
| `flow_mer_diversity.csv` | Flow-MER Waterbird Diversity (data.gov.au) |
| `flow_mer_breeding.csv` | Flow-MER Waterbird Breeding Colonies (data.gov.au) |
| `mdba_aws.csv` | MDBA 38-site AWS / SEA Waterbirds (data.gov.au) |
| `unsw_aerial.csv` | UNSW Aerial Survey 1983–2019 (Figshare DOI: 10.6084/m9.figshare.11853387) |

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw CSVs in data/raw/ (rename as above)

# 4. Run full pipeline
python run_pipeline.py

# 5. Skip plots (faster, headless)
python run_pipeline.py --skip-plots
```

## Outputs

| File | Description |
|---|---|
| `data/harmonised/mdb_waterbirds_master.parquet` | Full merged dataset (fast I/O) |
| `data/harmonised/mdb_waterbirds_master.csv` | Same, CSV format |
| `data/harmonised/macquarie_marshes.csv` | Macquarie Marshes subset |
| `data/harmonised/gwydir_wetlands.csv` | Gwydir Wetlands subset |
| `data/harmonised/mdb_all_sites.csv` | Full MDB subset |
| `outputs/plot1_abundance_over_time.png` | Annual counts by source |
| `outputs/plot2_species_richness.png` | Species richness at primary sites |
| `outputs/plot3_inundation_vs_abundance.png` | Inundation × abundance |
| `outputs/plot4_spatial_map.png` | Spatial distribution map |
| `outputs/qc_report.txt` | Data quality log (rows removed, issues flagged) |

## Unified Schema

Key columns in master output:

| Column | Description |
|---|---|
| `source_id` | S1/S2/S3/S4 (originating dataset) |
| `site_name` | Harmonised wetland/site name |
| `latitude` / `longitude` | Decimal degrees (validated) |
| `survey_date` | ISO 8601 date |
| `year` / `month` / `water_year` | Temporal indices |
| `scientific_name` | Normalised binomial (author stripped) |
| `abundance` | Individual count |
| `nest_count` / `brood_count` | Breeding indicators |
| `percent_full` | Wetland inundation % |
| `prop_surveyed` | Fraction of wetland sampled |
| `breeding_evidence` | Boolean |
| `survey_type` | aerial / ground / colony |

## Citation

- Kingsford et al. (2020). Australian Aerial Waterbird Survey Database. Figshare. https://doi.org/10.6084/m9.figshare.11853387
- CEWH (2024). Waterbird Diversity. Flow-MER Program. data.gov.au
- CEWH (2024). Waterbird Breeding Colonies. Flow-MER Program. data.gov.au
- MDBA/UNSW. Specified Environmental Assets Waterbird Survey. data.gov.au
