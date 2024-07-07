import multiprocessing as mp
import os
import re

import pyarrow.parquet as pq
import requests
import tqdm
import typer

app = typer.Typer()


def download(queue, retries: int, path: str):
    while True:
        data = queue.get()
        if data is None:
            break

        url, i = data
        url = str(url)
        idx = str(i)
        extension = re.search(r'\.([a-zA-Z0-9]+)$', url)
        if extension is None:
            print(f"Skipping {url} (unknown extension)")
            continue

        extension = extension.group(1)

        if extension is None:
            print(f"Skipping {url} (empty extension)")
            continue

        extension = extension.lower()

        if extension not in ("jpg", "jpeg", "png"):
            print(f"Skipping {url} (extension not in allowlist)")
            continue

        for i in range(retries):
            try:
                with requests.get(url, stream=True) as r, open(f"{path}/{idx}.{extension}", "wb") as f:
                    for chunk in r.iter_content(2 ** 20):
                        f.write(chunk)
            except KeyboardInterrupt:
                break
            except:
                continue
            break


def main(workers: int = 64, backlog: int = 32, table: str = "train_6.5.parquet", retries: int = 3,
         path: str = "./data"):
    mp.set_start_method("forkserver", force=True)
    queue = mp.Queue(backlog)
    downloaders = [mp.Process(target=download, args=(queue, retries, path), daemon=True) for _ in range(workers)]
    for d in downloaders:
        d.start()

    start = 0
    for f in os.listdir("./data"):
        try:
            idx = int(f.split(".")[-2])
        except KeyboardInterrupt:
            exit()
        except:
            continue
        start = max(idx, start)

    print(f"Restarting download at {start}")

    urls = pq.read_table(table)["URL"].combine_chunks()
    urls = urls[start:]

    try:
        for i, url in tqdm.tqdm(enumerate(urls, start)):
            queue.put((url, i))
    except KeyboardInterrupt:
        print("Shutting down gracefully")
        for _ in downloaders:
            queue.put(None)
