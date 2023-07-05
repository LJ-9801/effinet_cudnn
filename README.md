# cuDNN EfficientNet

## Requirements
1. Install cuDNN and cuBLAS
2. git clone https://github.com/llohse/libnpy to the include folder(this is for loading weights)
3. create a folder named "image" and put all image used for inference in there
4. enjoy Artificial Intelligence!


## Interface
Example run on a RTX 2060
```console
user@name:~/path/effinet_cudnn$ ./run.sh 
Enter path to weights folder:
/path/to/weights
Compiling and running...
Loading weights from file...
Done loading weights
Enter input image: chicken.jpg
Image size: (3 244 244)
Creating model...

Device Number: 0
  Device name: NVIDIA GeForce RTX 2060
  Memory Clock Rate (KHz): 5501000
  Memory Bus Width (bits): 192
  Peak Memory Bandwidth (GB/s): 264.048000

  Max Threads per Block: 1024
Copying weights to GPU...
Done copying weights

Initializing model...
Done initializing model

Start Inferencing...
Done Inferencing...
Inference time: 0.065585s
Predicted class: 7
```






