#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
// efficientnet
#include "include/efficientnet.h"

// file loading
void read_files(std::vector<file_info> &fn, std::string path);
bool compare(const file_info& a, const file_info& b);

// image utils
void read_image(std::string filename, std::vector<unsigned long> &img_size, std::vector<float> &img_data);


int main(int argc, char* argv[]){

    assert(argc == 2);
    std::string path = argv[1];

    std::vector<file_info> fn;
    
    std::cout << "Loading weights from file..." << std::endl;
    read_files(fn, path);
    std::cout << "Done loading weights" << std::endl;

    std::string input;
    std::cout << "Enter input image: ";
    std::cin >> input;
    input = "image/" + input;

    std::vector<unsigned long> img_size;
    std::vector<float> img_data;
    read_image(input, img_size, img_data);

    std::cout << "Creating model...\n" << std::endl;
    EfficientNet model = EfficientNet();
    
    std::cout << "Copying weights to GPU..." << std::endl;
    model.loadWeights(fn);  
    std::cout << "Done copying weights\n" << std::endl;
    
    std::cout << "Initializing model..." << std::endl;
    model.init();
    std::cout << "Done initializing model\n" << std::endl;

    DATA img = {img_data, img_size};

    model.forward(img);
    
    return 0;
}

// ============================ File Loading =======================================
void read_files(std::vector<file_info> &fn, std::string path){
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir (path.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file_name = ent->d_name;
            if(file_name.find(".npy") != std::string::npos){
                int end = file_name.find("_");
                int num = std::stoi(file_name.substr(0, end));
                std::string layer = file_name.substr(end+1, file_name.find("_", end+1) - end - 1);
                std::string function = file_name.substr(file_name.find("_", end+1) + 1, file_name.find(".npy") - file_name.find("_", end+1) - 1);
                file_info fi = {num, function, layer, file_name, path};
                fn.push_back(fi);
            }
        }
        closedir (dir);
    } else {
        std::cerr << "Could not open directory: " << path << std::endl;
        perror ("");
    }

    sort(fn.begin(), fn.end(), compare);
}

bool compare(const file_info& a, const file_info& b)
{
    return a.num < b.num;
}
// ==================================================================================


// ============================ Image Loading =========================================
void read_image(std::string filename, std::vector<unsigned long> &img_size, std::vector<float> &img_data){
    cv::Mat img = cv::imread(filename);
    cv::resize(img, img, cv::Size(244, 244));

    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};

    img_data = std::vector<float>(img.channels() * img.rows * img.cols);


    if (img.empty()){
        std::cout << "Could not read image" << std::endl;
        assert(false);
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < img.channels(); i++){
        for (int j = 0; j < img.rows; j++){
            for (int k = 0; k < img.cols; k++){
                float val = img.at<cv::Vec3b>(j, k)[i];
                val = (val / 255.0 - mean[i]) / std[i];
                img_data[i * img.rows * img.cols + j * img.cols + k] = val;
            }
        }
    }

    img_size = {(unsigned long)img.channels(), (unsigned long)img.rows, (unsigned long)img.cols};
    std::cout << "Image size: (";
    for (int i = 0; i < img_size.size(); i++){
        if (i == img_size.size() - 1)
            std::cout << img_size[i] << ")" << std::endl;
        else
            std::cout << img_size[i] << " ";
    }
}
// ====================================================================================


