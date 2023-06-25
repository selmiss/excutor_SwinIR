ARG JINA_VERSION=3.14.1

FROM jinaai/jina:${JINA_VERSION}-py38-standard

# copy will almost always invalid the cache
COPY . /workspace/
WORKDIR /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml", "--timeout-ready", "3600000"]