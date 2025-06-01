# aaa-nyc-llama-hackathon

A package to house the main github repo for the hackathon

## install
Setup the repo
```bash
git clone git@github.com:aaa-nyc-hackathon/aaa-nyc-llama-hackathon.git
cd aaa-nyc-llama-hackathon
pip install -r /path/to/requirements.txt
```

Install the react app dependencies
```bash
cd aaa_client
npm install  # install the
```

# Running the App

Running the app requires two steps, running the server on localhost and
running the client.

## Server on Localhost

```bash
python src/backend/fixed_app.py
```

## Client

Run the React app
```bash
npm run dev
```

Now you should have both running locally.