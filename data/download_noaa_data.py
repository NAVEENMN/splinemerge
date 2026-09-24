"""Download NOAA OI SST V2 monthly mean data and land-sea mask.

Downloads the sst.mnmean.nc and lsmask.nc files from the NOAA Physical
Sciences Laboratory (PSL) and stores them in data/noaa/ relative to the
project root.

The dataset is the NOAA Optimum Interpolation (OI) Sea Surface Temperature
(SST) V2, provided on a 1-degree global grid at monthly resolution from
1982 to present.  The land-sea mask (lsmask.nc) is needed to distinguish
ocean from land cells, since the SST product fills land cells by
interpolation.

Source: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
Direct download: https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc
Land-sea mask:   https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/lsmask.nc

Usage:
    python utils/download_noaa_data.py
"""

import os
import sys
import urllib.request

NOAA_URL = ("https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/"
            "sst.mnmean.nc")
LSMASK_URL = ("https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/"
              "lsmask.nc")


def download_file(url, dest_path, label):
    """Download a single file with progress reporting."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"File already exists: {dest_path} ({size_mb:.1f} MB)")
        print("To re-download, delete it first.")
        return True

    print(f"Downloading {label}...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest_path}")
    print()

    def progress_hook(block_num, block_size, total_size):
        """Report download progress."""
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path,
                                   reporthook=progress_hook)
        print()
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"Download complete: {dest_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"\nPlease download manually from:")
        print(f"  {url}")
        print(f"and place the file at:")
        print(f"  {dest_path}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dest_dir = os.path.join(project_root, 'data', 'noaa')
    os.makedirs(dest_dir, exist_ok=True)

    sst_path = os.path.join(dest_dir, 'sst.mnmean.nc')
    mask_path = os.path.join(dest_dir, 'lsmask.nc')

    ok_sst = download_file(NOAA_URL, sst_path,
                           "NOAA OI SST V2 monthly mean data")
    print()
    ok_mask = download_file(LSMASK_URL, mask_path,
                            "NOAA OI SST V2 land-sea mask")

    if not (ok_sst and ok_mask):
        sys.exit(1)


if __name__ == "__main__":
    main()
