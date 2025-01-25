FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m user
USER user

RUN pip install --user --no-cache-dir numpy pandas tensorflow scikit-learn pymatgen matplotlib seaborn bokeh tqdm

WORKDIR /work

USER user
CMD ["bash"]
