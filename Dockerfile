FROM ubuntu:22.04

RUN apt-get update
RUN apt-get -y install python3-pip curl jq gnupg software-properties-common

RUN add-apt-repository ppa:rmescandon/yq
RUN apt-get -y install yq

RUN useradd -m -U -u 1000 riptide
USER riptide

WORKDIR /home/riptide/

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.2.1
ENV PATH="/home/riptide/.local/bin:$PATH"
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project false &&\
    poetry install --no-interaction --no-ansi --only main --no-cache

COPY static/ static
COPY riptide/ riptide
COPY create_report.py .

ENTRYPOINT ["poetry", "run", "python", "create_report.py"]
