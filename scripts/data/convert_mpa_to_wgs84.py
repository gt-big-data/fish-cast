"""Convert CCAMLR MPA boundaries to plain longitude/latitude GeoJSON.

data/CCAMLR_MPA.json is in EPSG:6932 (a south-polar equal-area projection, metres),
not in degrees. Web maps (D3, OpenLayers) and most Python code expect EPSG:4326
longitude/latitude, so convert it once:

    python scripts/data/convert_mpa_to_wgs84.py
    # writes data/CCAMLR_MPA_wgs84.geojson
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/CCAMLR_MPA.json")
    p.add_argument("--output", default="data/CCAMLR_MPA_wgs84.geojson")
    a = p.parse_args()

    from pyproj import Transformer

    data = json.loads(Path(a.input).read_text())
    src = (data.get("crs") or {}).get("properties", {}).get("name", "EPSG:6932")
    src = "EPSG:" + src.split("::")[-1] if "::" in src else src
    to_wgs84 = Transformer.from_crs(src, "EPSG:4326", always_xy=True)

    def ring(coords):
        lon, lat = to_wgs84.transform([c[0] for c in coords], [c[1] for c in coords])
        return [[round(x, 5), round(y, 5)] for x, y in zip(lon, lat)]

    for f in data["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            g["coordinates"] = [ring(r) for r in g["coordinates"]]
        elif g["type"] == "MultiPolygon":
            g["coordinates"] = [[ring(r) for r in poly] for poly in g["coordinates"]]
        keep = ("GAR_Name", "GAR_Short_Label", "GAR_Start_Date", "GAR_Reference", "GAR_Size")
        f["properties"] = {k: f["properties"].get(k) for k in keep}
    data.pop("crs", None)
    Path(a.output).write_text(json.dumps(data))
    print(f"saved {a.output} ({len(data['features'])} protected areas, from {src})")


if __name__ == "__main__":
    main()
