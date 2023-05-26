ARG JINA_VERSION=3.14.1

FROM jinaai/jina:${JINA_VERSION}-py38-standard

RUN apt-get update && apt-get install -y --no-install-recommends git wget gcc \
&& apt-get clean && rm -rf /var/lib/apt/lists/*;

RUN python3 -m pip install --default-timeout=1000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# copy will almost always invalid the cache
COPY . /workspace/
WORKDIR /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml", "--timeout-ready", "3600000"]