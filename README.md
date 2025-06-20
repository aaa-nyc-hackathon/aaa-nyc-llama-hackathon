# aaa-nyc-llama-hackathon

A package to house the main github repo for the hackathon

## install
Setup the repo
```bash
git clone git@github.com:aaa-nyc-hackathon/aaa-nyc-llama-hackathon.git
cd aaa-nyc-llama-hackathon
# install python dependencies
pip install -r /path/to/requirements.txt
```

## Install the react app dependencies and setup the client
Read `aaa_client/client.md` for installation instructions on the client.

# Running the Backend Server
Running the app also requires running the server.

## Server on Localhost

```bash
python src/backend/fixed_app.py
```

Now you should have both running locally.

## Development

Additional setup for contribution to the repo is necessary.
You should first ensure you are able to run both the client and the
server code locally and execute a simple video processing workflow
via the locally running app. Once you can do these steps the next step
is to setup development env.

## Dev Dependencies

These are dependencies not necessary for running the application but necessary
for development. They typically include items used for testing code and
workflows.

```bash
pip install -r ./dev-requirements.txt
```

## development workflow

We use [ruff](https://docs.astral.sh/ruff/) for the good stuff.

The typical workflow looks like:
... hack hack hack ...
```bash
ruff check --fix # lint the changes
ruff format # format changes
```
Note: `ruff` is installed from the `dev-requirements.txt` file so you
will already have the tool in your env if you followed above instructions.

For the `format` command you can pass `--check` to see which files would
be formatted before applying the changes.

We may add linting and format exceptions to ruff as we continue development.
