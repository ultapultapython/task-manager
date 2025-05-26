import requests
import time
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate


URLS = [
    "https://www.python.org/",
    "https://fastapi.tiangolo.com/",
    "https://pypi.org/",
    "https://www.github.com/",
    "https://stackoverflow.com/",
    "https://www.wikipedia.org/",
    "https://www.reddit.com/",
    "https://www.nytimes.com/",
]

def download_url(url: str) -> int:
    """Download content from url and return content size in bytes."""
    response = requests.get(url)
    return len(response.content)

def sequential_download(urls):
    results = []
    start = time.perf_counter()
    for url in urls:
        size = download_url(url)
        results.append((url, size))
    duration = time.perf_counter() - start
    return results, duration

def threaded_download(urls, max_workers=5):
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_url, url) for url in urls]
        for url, future in zip(urls, futures):
            size = future.result()
            results.append((url, size))
    duration = time.perf_counter() - start
    return results, duration

def main():
    print("Starting sequential download...")
    seq_results, seq_time = sequential_download(URLS)
    print(f"Sequential download took {seq_time:.2f} seconds.\n")

    print("Starting threaded download...")
    thread_results, thread_time = threaded_download(URLS)
    print(f"Threaded download took {thread_time:.2f} seconds.\n")

    # Summary Table
    table = [
        ["Method", "Total Time (s)", "Time Saved (s)", "Speedup Factor"],
        [
            "Sequential",
            f"{seq_time:.2f}",
            "-",
            "-"
        ],
        [
            "ThreadPoolExecutor",
            f"{thread_time:.2f}",
            f"{seq_time - thread_time:.2f}",
            f"{seq_time / thread_time:.2f}"
        ]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="grid"))

if __name__ == "__main__":
    main()
