GPU="1,3"


echo "Training pipeline started!"



cmd="python train.py --gpu ${GPU} --config ./configs/SCD_SECOND.yaml"
echo $cmd
eval $cmd
sleep 10


cmd="python train.py --gpu ${GPU} --config ./configs/SCD_SECOND.yaml"
echo $cmd
eval $cmd
sleep 10


cmd="python train.py --gpu ${GPU} --config ./configs/SCD_SECOND.yaml"
echo $cmd
eval $cmd
sleep 10


cmd="python train.py --gpu ${GPU} --config ./configs/SCD_SECOND.yaml"
echo $cmd
eval $cmd
sleep 10



echo "Training pipeline ends!"