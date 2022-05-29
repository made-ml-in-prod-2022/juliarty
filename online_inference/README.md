**This is the second homework.**


To build docker image run:

```
docker build -t juliarty/online_inference .
```

To pull the latest official image run:

```
docker pull juliarty/online_inference:latest
```

To run a container run:

```
docker run -p 9999:9999 -e MODEL_GDRIVE_URL=https://drive.google.com/uc?id=1YY9Vo5wxwUOa_42t52zYL4to-wxJtqIY juliarty/online_inference
```

To test service run:

```
python -m test_service --host 0.0.0.0 --port 9999 --req-num 20
```
