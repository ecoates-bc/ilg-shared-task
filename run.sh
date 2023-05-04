docker build -t ilg .
docker run --gpus all -it ilg python3 -m GlossingLSTM