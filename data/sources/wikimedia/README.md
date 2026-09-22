# Wikimedia Commons Data Pipeline

This data source fetches giant river otter images, metadata, descriptions, page links, and licensing information from [Wikimedia Commons](https://commons.wikimedia.org/).

## Workflow

### 1. Fetch Search Results

Search Wikimedia Commons and save result snapshots:

```sh
uv run data/sources/wikimedia/search_wikimedia.py --query '"giant river otter"'
uv run data/sources/wikimedia/search_wikimedia.py --query '"Pteronura brasiliensis"'
```

This creates JSON snapshots:
- `data/sources/wikimedia/search_results_giant-river-otter.json`
- `data/sources/wikimedia/search_results_pteronura-brasiliensis.json`

### 2. Download Images and Generate Metadata CSV

Download high-resolution image files and generate `wikimedia_results.csv`:

```sh
uv run data/sources/wikimedia/download_wikimedia_images.py data/sources/wikimedia/search_results_giant-river-otter.json data/sources/wikimedia/search_results_pteronura-brasiliensis.json
```

