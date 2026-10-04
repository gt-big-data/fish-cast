# Data Visualization

This folder will hold the FishCast web app and the static figure pipeline.

Planned layout (settle in weeks 1-2):

- `viz/app/` front end (TypeScript, D3 geographic projections for the south-polar map)
- `viz/api/` data API (FastAPI) serving predictions, time series and benchmark tables
- `viz/figures/` scripts that regenerate every static figure with one command
- `viz/STYLE_GUIDE.md` projection, colour maps, fonts, units

Before choosing a map library, build a small prototype that draws the Antarctic coastline in a south-polar stereographic projection. Leaflet and MapLibre use Web Mercator and are poor fits for the Southern Ocean.
