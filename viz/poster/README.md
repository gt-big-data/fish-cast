# Demo Day poster + interactive map

## Live demo (scan the QR)

**https://aye1024.github.io/fish-cast/**

Deployed from this folder via GitHub Pages (workflow: `.github/workflows/deploy-demo-pages.yml`).

## Run locally

```bash
python -m http.server 8765 --directory viz/poster
```

Open http://127.0.0.1:8765/demo.html

Data lives in `viz/poster/data/` (`hauls.json`, `mpa.geojson`) so the page is self-contained.

## Controls

| Control | What it does |
| --- | --- |
| Year sliders / decade chips | Filter hauls by year |
| Show zeros | Include / hide empty nets |
| Show MPAs | Toggle CCAMLR polygons |
| Hotspots only | Keep ~top 10% densities |
| Min density | Log-scale density floor |
| Color by | Density (viridis) or presence-only |
| Projection | Equal-area vs stereographic |
| Rotate longitude | Spin the polar view |
| Reset / Peninsula / Ross Sea | View presets |
| Drag / scroll | Pan and zoom |
| Click a point | Detail panel (date, density, SST, ice, salinity) |

Live stats on the right update with every filter.

## Poster assets

| File | Use |
| --- | --- |
| `figures/qr-github.png` | Top-right QR → live demo URL above |
| `figures/demo-running.png` | Demo panel screenshot |
| `figures/polar_map.png` / `data-map.png` | Data panel figure |
