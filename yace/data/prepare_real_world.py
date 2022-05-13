import os

from pathlib import Path
from timeit import default_timer as timer

import click
import numpy as np
import requests

from scipy.sparse import csr_matrix
from tqdm import tqdm


def load_census_data_set(file_path: str):
    print(f"Loading Census data from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=1,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    return data[:,1:]


def load_tower_dataset(file_path: str):
    print(f"Loading Tower dataset from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=0,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    
    D = 3
    N = int(data.shape[0] / D)
    return data.reshape((N, D))


def load_covertype_dataset(file_path: str):
    print(f"Loading Covertype dataset from {file_path}...")
    start_time = timer()
    data = np.loadtxt(
        fname=file_path,
        dtype=np.double,
        delimiter=",",
        skiprows=0,
        unpack=False
    )
    end_time = timer()
    print(f"Loaded in {end_time - start_time:.2f} secs")
    return data[:, 0:-1] # Skip the last column which is the classification column


def download_file(url: str, file_path: Path, file_size: int=None):
    """Downloads file from `url` to `file_path`."""
    print(f"Downloading {url} to {file_path}...")
    chunk_size = 1024
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('Content-Length', file_size))
    file_size = chunk_size * 10 if file_size is None else file_size
    with open(file_path, 'wb') as f:
        pbar = tqdm(unit="B", unit_scale=True, total=file_size)
        for chunk in r.iter_content(chunk_size=chunk_size): 
            if chunk: # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


class RealWorldDataSetsPrep:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        if not self._output_dir.exists():
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self._prepare_census()
        self._prepare_tower()
        self._prepare_covertype()

    def _prepare_data_set(self, data_set_name: str, download_url: str, file_size: int, data_loader_fn):
        final_file_path = self._output_dir / f"{data_set_name}.npz"
        if final_file_path.exists():
            print(f"Data set '{data_set_name}' is already prepared. Skipping...")
            return
        local_file_name = os.path.basename(download_url)
        local_file_path = self._output_dir / local_file_name
        if not local_file_path.exists():
            download_file(
                url=download_url,
                file_path=local_file_path,
                file_size=file_size
            )
        data = data_loader_fn(local_file_path)
        np.savez_compressed(final_file_path, data=data)
        os.remove(local_file_path)
        print(f"Preparation of the {data_set_name} data set is complete.")

    def _prepare_census(self):
        self._prepare_data_set(
            data_set_name="census",
            download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt",
            file_size=361344227,
            data_loader_fn=load_census_data_set,
        )

    def _prepare_tower(self):
        self._prepare_data_set(
            data_set_name="tower",
            download_url="http://homepages.uni-paderborn.de/frahling/instances/Tower.txt",
            file_size=52828754,
            data_loader_fn=load_tower_dataset,
        )

    def _prepare_covertype(self):
        self._prepare_data_set(
            data_set_name="covertype",
            download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
            file_size=11240707,
            data_loader_fn=load_covertype_dataset,
        )


@click.command(help="Prepare real-world data sets.")
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=False,
    default="data/input",
)
def main(output_dir: str):
    RealWorldDataSetsPrep(
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
