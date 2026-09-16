# Flickr data

To update the Flickr images:

Requires a Flickr API key. Put `FLICKR_API_KEY=your-key` in `data/sources/flickr/.env`.

Fetch the search results (defaults to open-licensed images only:

```sh
uv run data/sources/flickr/search_flickr.py --query "giant river otter"
uv run data/sources/flickr/search_flickr.py --query "Pteronura brasiliensis"
```

Download the largest available images from each result set:

```sh
uv run data/sources/flickr/download_flickr_images.py data/sources/flickr/search_results_giant-river-otter.json
uv run data/sources/flickr/download_flickr_images.py data/sources/flickr/search_results_pteronura-brasiliensis.json
```

This updates `data/sources/flickr/flickr_results.csv`.
Encounter IDs are derived by `data/sources/flickr/flickr_encounters.py` from
the Flickr owner and capture time.

Many images do not contain otters. Filter the downloaded images with the otter detector:

```sh
uv run data/sources/flickr/filter_detected_otters.py --device 0
```

This writes `data/sources/flickr/flickr_results_filtered.csv`.
