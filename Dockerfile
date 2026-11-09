# Prototype container (CPU inference). EXPERIMENTAL: not tested yet; the supported ways to
# run the prototype are notebook 03 (Colab) or a local install (see README).
#   docker build -t s1lulc .
#   docker run --rm -v ~/.config/earthengine:/root/.config/earthengine \
#     -v $PWD/bundle:/bundle -v $PWD/out:/out s1lulc \
#     predict --lat 48.14 --lon 11.42 --start 2025-10 --bundle /bundle \
#     --project YOUR-EE-PROJECT --out /out
# Earth Engine credentials come from the host (run `earthengine authenticate` once).
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md requirements.txt ./
COPY configs configs
COPY s1lulc s1lulc
RUN pip install -e ".[gee,geo]" segmentation-models-pytorch matplotlib

ENTRYPOINT ["s1lulc"]
CMD ["--help"]
