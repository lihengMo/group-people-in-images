# Multi grouping people in images

## Environment installation

1. **Create conda**
   ```
   conda create -n venv pip python=3.5  # select python version, here I use 3.5
   ```

2. **Install mxnet.**
   ```
   pip install requests==2.18.4
   pip install mxnet-cu90
   ```

3. **Install tensorflow and keras. I use the keras/tensorflow implement of yolo to do the head detection. if there is enough time it is better to use a mxnet version of yolo.** 
   ```
   pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.9.0-cp35-cp35m-linux_x86_64.whl
   pip install keras
   ```

4. **Python packages might missing:** cython, opencv-python, scikit-learn, imgaug, pickle,h5py, PyYAML,tqdm. By using the requirements.txt or just use 'pip' to install is ok
   ```
   pip install -r requirements.txt
   ```
   ***be careful that the version of scikit-learn should be 0.19.0, or it will cause problem while using k-means***

## Training/Testing Pair-net model and clustering


1. **for the dataset:**

	```
	./data/GroupActivityPerson/
	```

2. **for the pretrained model of resnet is under:**
	```
	./model/pretrained_model/resnet-50-0000.params
	```

3. **running the Training of the Pair-Net**
    ```
    python train_resnet.py
    ```
    A cache folder would be created automatically to save the dataset loading file `soa_2018_trainval_roidb.pkl` under `data/cache/`.
    
    for head detection:
    Now there is still a problem that because I use the keras version of yolo, so the head detection can not be run with the Pair-Net at the same time, which will cause some cudnn memory problems. so I will firstly do the head detection on the dataset and save the detecting result `roi_head.pkl` under the `data/cache/`. 

    for the yolo model using for head detection, make sure the model file is under:
    ```
    ./headdet/config.json
    ./headdet/model.h5
    ./headdet/full_yolo_backend.h5
    ``` 

    The Pair-Net model will be saved under the `model/` after every epoch.

    the training log`train_log_XX.log` will be automatically saved under the `./output/`

4. **running the Testing of the Pair-Net and clustering**
   ```
   python test_resnet.py
   ```
   
   the same as training process, it will create testset pkl file and testset head detection file under the `data/cache/`.

   put the test model under:
   ```
   ./model/test/
   ```

   for group clustering, please check the clustering parameter in the `test_resnet.py`:
   ```
   do_cluster = True                 #whether do clustering
   given_cluster_num = True          #whether given cluster number
   cut_matrix_threshold = 0.45       #if using pairwise matrix cut, set a threshold between (0,1) ,if not using please set threshold=0.
   vis = True                        #whether visualize the cluster result
   ```

   the pairwise relation prediction result log`result_log_XX.log` will be automatically saved under the `./output/`
   
   if doing the cluster and choose visualize the cluster result will be automatically saved under the `./cluster_result/`
      
5. **Please find more details of the parameter setting in the code.**



