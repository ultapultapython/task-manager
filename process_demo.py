import os
import time
import requests
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate


TEMP_DIR = "temp_images"
PROCESSED_DIR = "processed_images"


IMAGE_URLS = [
    "https://plus.unsplash.com/premium_photo-1721317368393-09204f05aab5",
    "https://images.unsplash.com/photo-1709884735017-114f4a31f944",
    "https://images.unsplash.com/photo-1546464677-c25cd52c470b",
    "https://plus.unsplash.com/premium_photo-1667538960183-82690c60a2a5",
    "https://images.unsplash.com/photo-1616986035206-90bc396686c7",
    "https://images.unsplash.com/photo-1679908731995-4ee6f1ceedbc",
    "https://images.unsplash.com/photo-1617079104500-1324836c5a8e",
    "https://images.unsplash.com/photo-1720519118474-2acfc6198fa9",
    "https://images.unsplash.com/photo-1545426373-6588267475be"
]

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return os.path.basename(save_path), "Downloaded"
    except Exception as e:
        return os.path.basename(save_path), f"Error: {str(e)}"

def process_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((800, 800), resample=Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(3))

        # Save processed image
        base_name = os.path.basename(image_path)
        output_path = os.path.join(PROCESSED_DIR, base_name)
        img.save(output_path)

        return base_name, "Processed"
    except Exception as e:
        return os.path.basename(image_path), f"Error: {str(e)}"

def delete_image(image_path):
    try:
        os.remove(image_path)
        return os.path.basename(image_path), "Deleted"
    except Exception as e:
        return os.path.basename(image_path), f"Error: {str(e)}"

def sequential_execution(image_urls):
    results = []
    start_time = time.time()
    for url in image_urls:
        file_name = os.path.basename(url.split("?")[0]) + ".jpg"
        download_path = os.path.join(TEMP_DIR, file_name)
        results.append(download_image(url, download_path))
        results.append(process_image(download_path))
        results.append(delete_image(download_path))
    elapsed = time.time() - start_time
    return results, elapsed

def parallel_execution(image_urls):
    results = []
    start_time = time.time()
    download_paths = [os.path.join(TEMP_DIR, os.path.basename(url.split("?")[0]) + ".jpg") for url in image_urls]

    with ProcessPoolExecutor() as executor:
        download_results = list(executor.map(download_image, image_urls, download_paths))
        results.extend(download_results)

        process_results = list(executor.map(process_image, download_paths))
        results.extend(process_results)

        delete_results = list(executor.map(delete_image, download_paths))
        results.extend(delete_results)

    elapsed = time.time() - start_time
    return results, elapsed

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Starting Sequential Execution...\n")
    seq_results, seq_time = sequential_execution(IMAGE_URLS)

    print("Starting Parallel Execution...\n")
    par_results, par_time = parallel_execution(IMAGE_URLS)

    # Compare performance
    table = [
        ["Sequential", f"{seq_time:.2f} seconds"],
        ["Parallel (ProcessPoolExecutor)", f"{par_time:.2f} seconds"]
    ]
    print(tabulate(table, headers=["Execution Type", "Time Taken"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
