
#!/bin/bash
nvcc -O3 utils.cu efficientnet.cpp main.cpp -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcudnn -lcublas -lcudart -o effnet && ./effnet /home/jg000/resnet/cudnn_effnet/weights
rm effnet