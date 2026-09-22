# iNaturalist data

This source uses the unauthenticated public iNaturalist API. The default taxon
is Giant Otter (*Pteronura brasiliensis*, iNaturalist taxon `41845`).

Fetch public observations with licensed photos (defaults to open-licensed images only; use `--all-licenses` to include unlicensed/all-rights-reserved):

```sh
uv run data/sources/inat/fetch_inaturalist_observations.py
```

The fetch writes:

- `inaturalist_results.csv`: normalized observation metadata, including quality
  grade, observation license, public coordinates, observer, and photo IDs.
- `inaturalist_photos.csv`: one row per photo, including its separate license,
  attribution, dimensions, and original-size URL.
- `inaturalist_observations.jsonl.gz`: the complete API observation objects.
- `snapshot.json`: retrieval provenance and counts by quality grade and license.
- `data/results.csv`: unified dataset automatically synchronized with the new observations and photo metadata.

No API key or login is required. Coordinates are the public coordinates exposed
by iNaturalist and may be obscured.

Download the images with the downloader (defaults to open-licensed photos only; use `--all-licenses` to include unlicensed photos):

```sh
uv run data/sources/inat/download_inaturalist_images.py \
  data/sources/inat/inaturalist_results.csv
```

When `inaturalist_photos.csv` is beside the observation CSV, the downloader
uses its saved original URLs automatically (skipping non-open licenses by default).
This avoids refetching every observation from the API and still checks the shared
tree before opening an image URL. When new images are saved, the downloader updates
the local image paths in `data/results.csv`.

To manually rebuild or synchronize `data/results.csv` from all data sources at any time:

```sh
uv run data/sources/build_results.py
```

The API and image downloader both wait at least one second between observation
requests. Preserve the observation and photo license and attribution columns
when using downloaded material.

