# Coreset Evaluation

A Python package to evaluate coreset constructions.

## Getting Started

This projects relies on [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/).

1. Install the required Python version:

   ```bash
   pyenv install
   ```

2. Install dependencies

   ```bash
   poetry install --no-dev
   ```

3. Create jobs

   ```bash
   poetry run python -m yace.create_jobs -q data/queue -o data/experiments -t simple -n initial_ss_01 -r 1
   ```

4. Run worker

   ```bash
   poetry run python -m yace.run_worker -q data/queue --n-threads 1 --max-active 1
   ```

5. Compute distortions

   ```bash
   poetry run python -m yace.calc_distortions -r data/experiments --n-jobs 1 --n-threads 1
   ```

6. Download experiments results from server

   ```bash
   rsync -av skadi:/home/omar/code/yace/data/experiments/ data/experiments-skadi
   ```
