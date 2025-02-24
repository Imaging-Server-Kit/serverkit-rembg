![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# Rembg API in docker

Implementation of a web API server for [rembg](https://github.com/danielgatis/rembg).

## Installation with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Running tests:

```
pytest
```

## Installation with `docker`

Build the docker image:

```
docker build -t serverkit-rembg .
```

Run the server in a container:

```
docker run -it --rm -p 8000:8000 serverkit-rembg:latest
```

Running tests:

```
docker run --rm serverkit-rembg:latest pytest
```

Pushing the image to `registry.rcp.epfl.ch`:

```
docker tag serverkit-rembg registry.rcp.epfl.ch/imaging-server-kit/serverkit-rembg
docker push registry.rcp.epfl.ch/imaging-server-kit/serverkit-rembg
```