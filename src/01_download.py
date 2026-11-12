"""
Download the TTC subway delay files from Toronto's open data portal.

Run it with:
    python src/01_download.py

Safe to run twice: it skips files it already has. Everything lands in
data/raw/ and is never edited after this point.

HOW THIS WORKS
--------------
The portal runs on CKAN, which is open-source dataset-hosting software. CKAN
exposes a JSON API. We ask it one question:

    "describe the package called ttc-subway-delay-data"

and it answers with a JSON object. Inside that object is a list called
"resources" - one entry per downloadable file, each with a name, a format
("XLSX", "CSV", ...) and a direct download url. So instead of clicking
twelve download links in a browser, we loop over that list.

IF THIS SCRIPT FAILS
--------------------
The API hostname in config.py (CKAN_BASE) is the one the portal has used,
but hostnames change and I could not verify it for you. If you get a
connection error or a 404:

  1. Open https://open.toronto.ca/dataset/ttc-subway-delay-data/
  2. Find the developers / API section on that page.
  3. Copy the base URL it shows and paste it into CKAN_BASE in config.py.

If it still fights you after twenty minutes, download the files by hand from
that page into data/raw/ and move on. Script 02 does not care how the files
got there. Come back and finish this on Day 5.
"""

import sys

import requests

from config import CKAN_BASE, PACKAGE_ID, RAW_DIR


# Formats we want. The delay data itself is XLSX; the readme and the delay
# code lookup may be XLSX or CSV, and we want those too.
WANTED_FORMATS = {"XLSX", "XLS", "CSV"}


def fetch_resource_list():
    """Ask CKAN to describe the package, return its list of resources.

    Returns a list of dicts. Each dict has at least 'name', 'format', 'url'.
    """
    url = f"{CKAN_BASE}/package_show"

    # params= lets requests build the query string for us, correctly escaped.
    # timeout= means a dead server fails in 30 seconds instead of hanging.
    response = requests.get(url, params={"id": PACKAGE_ID}, timeout=30)

    # Raises an exception on a 404 or 500 so we fail loudly, not silently.
    response.raise_for_status()

    payload = response.json()

    # CKAN wraps everything in {"success": true, "result": {...}}.
    if not payload.get("success"):
        raise RuntimeError(f"CKAN said the request failed: {payload}")

    return payload["result"]["resources"]


def download_one(resource):
    """Download a single resource into RAW_DIR. Skip it if already present.

    Returns a short status string for the summary table.
    """
    name = resource.get("name", "unnamed")
    fmt = (resource.get("format") or "").upper()
    url = resource.get("url")

    if fmt not in WANTED_FORMATS:
        return "skipped (format)"
    if not url:
        return "skipped (no url)"

    # Build a safe filename. Resource names can contain spaces and slashes.
    safe_name = name.replace("/", "-").replace(" ", "_")
    extension = fmt.lower()
    if not safe_name.lower().endswith(f".{extension}"):
        safe_name = f"{safe_name}.{extension}"

    destination = RAW_DIR / safe_name

    if destination.exists():
        return f"already have ({destination.stat().st_size:,} bytes)"

    # stream=True downloads in chunks instead of loading the whole file into
    # memory. These files are tens of megabytes, so it matters a little.
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return f"downloaded ({destination.stat().st_size:,} bytes)"


def main():
    # parents=True creates data/ as well if it is missing.
    # exist_ok=True means running twice is not an error.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Asking {CKAN_BASE} about '{PACKAGE_ID}'...\n")

    try:
        resources = fetch_resource_list()
    except Exception as error:
        print(f"Could not reach the portal API: {error}\n")
        print("Read the 'IF THIS SCRIPT FAILS' note at the top of this file.")
        sys.exit(1)

    print(f"The package lists {len(resources)} resources.\n")

    for resource in resources:
        name = resource.get("name", "unnamed")
        fmt = (resource.get("format") or "?").upper()
        status = download_one(resource)
        print(f"  {name:<45} {fmt:<6} {status}")

    # Count what we ended up with, so the script tells you where you stand.
    files = sorted(RAW_DIR.iterdir())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())

    print(f"\ndata/raw/ now holds {len(files)} files, {total_bytes / 1e6:.1f} MB")
    print("\nNext: python src/02_build.py")


if __name__ == "__main__":
    main()
