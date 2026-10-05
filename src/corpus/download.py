# -*- coding: utf-8 -*-
"""Download dated CC-News WARC archives."""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


BASE_URL = "https://data.commoncrawl.org/crawl-data/CC-NEWS/"
DOWNLOAD_URL = "https://data.commoncrawl.org/"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_date_arg(date_str: str) -> tuple[str, str | None]:
    """Return (YYYY/MM, optional YYYYMMDD filter) from a month or day argument."""
    parts = date_str.strip("/").split("/")
    if len(parts) == 2:
        year_month = f"{parts[0]}/{int(parts[1]):02d}"
        return year_month, None
    if len(parts) == 3:
        year_month = f"{parts[0]}/{int(parts[1]):02d}"
        day_filter = f"{parts[0]}{int(parts[1]):02d}{int(parts[2]):02d}"
        return year_month, day_filter
    raise SystemExit("Invalid date. Use YYYY/MM or YYYY/MM/DD.")


def warc_paths_url(year_month: str) -> str:
    return BASE_URL + f"{year_month}/warc.paths.gz"


def filter_paths_by_day(paths: list[str], day_filter: str | None) -> list[str]:
    if day_filter is None:
        return paths
    needle = f"CC-NEWS-{day_filter}"
    return [p for p in paths if needle in os.path.basename(p)]


def read_paths_gz(path: str) -> list[str]:
    with gzip.open(path, "rt") as f:
        return [line.strip() for line in f if line.strip()]


def _fetch_warc_paths_simple(local_path: str, year_month: str) -> None:
    """Fetch the WARC path list once for size estimation."""
    url = warc_paths_url(year_month)
    logging.info(f"Fetching {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fd.write(chunk)


def _head_size(args_tuple: tuple[int, str]) -> tuple[int, str, float]:
    i, path = args_tuple
    url = DOWNLOAD_URL + path
    resp = requests.head(url, timeout=30)
    size = int(resp.headers.get("Content-Length", 0))
    return i, path, size / (1024 * 1024)


def cmd_size(args: argparse.Namespace) -> None:
    year_month, day_filter = parse_date_arg(args.date)

    with tempfile.TemporaryDirectory() as tmp:
        paths_local = os.path.join(tmp, "warc.paths.gz")
        _fetch_warc_paths_simple(paths_local, year_month)
        file_paths = read_paths_gz(paths_local)

    if day_filter is not None:
        file_paths = filter_paths_by_day(file_paths, day_filter)
        if not file_paths:
            raise SystemExit("No WARC files match that day.")

    logging.info(f"Checking size of {len(file_paths)} files with {args.workers} workers.")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(_head_size, enumerate(file_paths)),
            total=len(file_paths), desc="Checking sizes",
        ))

    total_mb = 0.0
    for i, path, mb in sorted(results):
        print(f"{i+1}. {path} — {mb:.2f} MB")
        total_mb += mb

    print(f"\nTotal: {total_mb:.2f} MB  ({total_mb / 1024:.2f} GB)")


def make_pooled_session(workers: int) -> requests.Session:
    """Create a shared session with a connection pool sized for the workers."""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_with_retries(
    session: requests.Session,
    url: str,
    local_path: str,
    chunk_bytes: int,
    retries: int = 5,
    backoff: int = 10,
) -> bool:
    wait_time = backoff
    tmp_path = local_path + ".part"
    for _attempt in range(retries):
        try:
            logging.info(f"Downloading: {url}")
            with session.get(url, stream=True, timeout=(15, 300)) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as fd:
                    shutil.copyfileobj(response.raw, fd, length=chunk_bytes)
            os.replace(tmp_path, local_path)
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error downloading {url}: {e}. Retry in {wait_time}s...")
            time.sleep(wait_time)
            wait_time *= 2
    logging.error(f"Failed to download {url} after {retries} attempts.")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def cmd_crawl(args: argparse.Namespace) -> None:
    year_month, day_filter = parse_date_arg(args.date)

    download_folder = os.path.join(args.out_dir, year_month.replace("/", "-"))
    os.makedirs(download_folder, exist_ok=True)
    chunk_bytes = args.chunk_mb * 1024 * 1024

    session = make_pooled_session(args.workers)

    warc_paths_local = os.path.join(download_folder, "warc.paths.gz")
    if not os.path.exists(warc_paths_local):
        if not download_with_retries(
            session, warc_paths_url(year_month), warc_paths_local, chunk_bytes
        ):
            raise SystemExit("Failed to download warc.paths.gz.")

    logging.info(f"Reading WARC paths from: {warc_paths_local}")
    file_paths = read_paths_gz(warc_paths_local)

    if day_filter is not None:
        before = len(file_paths)
        file_paths = filter_paths_by_day(file_paths, day_filter)
        logging.info(f"Day filter {day_filter}: {len(file_paths)} of {before} files match.")
        if not file_paths:
            raise SystemExit("No WARC files match that day — check the date.")

    def download_one(path: str) -> None:
        url = DOWNLOAD_URL + path
        local_filename = os.path.join(download_folder, os.path.basename(path))
        if os.path.exists(local_filename):
            logging.info(f"File exists, skipping: {local_filename}")
            return
        download_with_retries(session, url, local_filename, chunk_bytes)

    logging.info(f"Starting download of {len(file_paths)} WARC files with {args.workers} workers.")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(
            executor.map(download_one, file_paths),
            total=len(file_paths), desc="Downloading WARC files",
        ))

    if os.path.exists(warc_paths_local):
        os.remove(warc_paths_local)

    logging.info("All files downloaded.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CC-NEWS download utilities (size estimation and full crawl)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_size = sub.add_parser("size", help="Estimate total download size for a month/day.")
    p_size.add_argument("date", type=str, help="YYYY/MM or YYYY/MM/DD.")
    p_size.add_argument("--workers", type=int, default=20)
    p_size.set_defaults(func=cmd_size)

    p_crawl = sub.add_parser("crawl", help="Download CC-NEWS WARC files for a month/day.")
    p_crawl.add_argument(
        "--date", type=str, required=True,
        help="YYYY/MM for a whole month, or YYYY/MM/DD for a single day.",
    )
    p_crawl.add_argument(
        "--out-dir", type=str, required=True,
        help="Base output directory. A YYYY-MM subfolder is created under it.",
    )
    p_crawl.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Parallel download workers.",
    )
    p_crawl.add_argument(
        "--chunk-mb", type=int, default=1, help="Write chunk size in MiB.",
    )
    p_crawl.set_defaults(func=cmd_crawl)

    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
