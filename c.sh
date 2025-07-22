#!/usr/bin/env bash

python setup.py build_ext --inplace



for minimum in 5; do # 10 20; do
  for model in desenet201; do # do mobilenetv2 resnet50v2; do
  # for model in vgg19 efficientnetB3 vit_large vgg16 denset; do # do mobilenetv2 resnet50v2; do
    for dataset in pr; do
      for size in 512; do # 400 512; do
        for color in RGB; do # RGB; do
          for smote in 0 30 60 90 120; do # RGB; do
            for classifier in MLPClassifier SVC; do # RandomForestClassifier KNeighborsClassifier MLPClassifier SVC; do #DecisionTreeClassifier RandomForestClassifier KNeighborsClassifier MLPClassifier SVC
              python main.py -i /home/msechineli/Documentos/TCC/features/${dataset}+${minimum}/${color}/${size}/${model}/features -s ${smote} -c ${classifier} -o /home/msechineli/Documentos/TCC/smote-2/${dataset}+${minimum}+${color}+${size}+${model}+${classifier}+SMOTE+${smote}
              # python main.py -i /home/msechineli/Documentos/TCC/pr/original/features/RGB/512/5/vgg16/pr_dataset+col=RGB+size=512+min=5+cnn=vgg16 -c ${classifier} -o /home/msechineli/Documentos/TCC/smote/${dataset}+${minimum}+${color}+${size}+${model}+${classifier}-rrrr
            done
          done
        done
      done
    done
  done
done

#DATASET=regions_dataset
#for minimum in 5 10 20; do
#  for model in resnet50v2 vgg16 mobilenetv2; do  
#    for region in North Northeast Middlewest South Southeast; do
#      for size in 256 400 512; do
#        python main.py --contrast 1.2 --formats npz --height ${size} -i ./datasetv2/new/formatted/${DATASET}+${region}+${minimum}/${color}/${size}/ -o media/xandao/eec07521-c36a-4d2b-9047-0110e7749eae/datasetv2/new/features/${DATASET}+${region}+${minimum}/${color}/${size}/${model} --patches 3 --width ${size} --save_images --model ${model} --orientation horizontal
#      done
#    done
#  done
#done
