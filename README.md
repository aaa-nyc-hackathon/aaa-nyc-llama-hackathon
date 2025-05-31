# aaa-nyc-llama-hackathon

A package to house the main github repo for the hackathon

## install

```bash
git clone git@github.com:aaa-nyc-hackathon/aaa-nyc-llama-hackathon.git
cd aaa-nyc-llama-hackathon
pip install -r /path/to/requirements.txt.
```

## then run the cli tool

```bash
python extract_actions.py
```

## make a curl request against the localhost
```bash
curl -X POST "http://localhost:8000/api/analyze-video"  \
     -H "accept: application/json"  \
     -H "Content-Type: multipart/form-data" \
     -F "file=@aaa-nyc-llama-hackathons/videos/alabama_clemson_30s_clip.mp4;type=video/mp4" \
     --verbose
```


