pkill -f python3.10
pkill -f tensorboard

# wait 2 seconds
sleep 2

rm -rf out.log
nohup python3.10 -u train.py >> out.log 2>&1 &
tensorboard --logdir logs --port 6006 --bind_all &
tail -f out.log