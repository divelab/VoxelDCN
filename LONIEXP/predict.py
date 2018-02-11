import os
gpu = 10
os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python main.py --option=predict")
for i in range(2,21):
    os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python main"+str(i)+".py --option=predict")
