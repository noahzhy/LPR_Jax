pkill -f python3.9
pkill -f tensorboard

# wait 2 seconds
sleep 2

nohup python3.9 -u train.py > out.log 2>&1 &
tensorboard --logdir logs --port 6006 --bind_all &
