
#!/bin/bash
echo "Enter path to weights folder:"
read path
echo "Compiling and running..."
nvcc -O3 utils.cu efficientnet.cpp main.cpp -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcudnn -lcublas -lcudart -o effnet && ./effnet $path
rm effnet