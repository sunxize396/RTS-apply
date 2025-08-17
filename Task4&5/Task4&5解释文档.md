





# 一. 结构要求
      项目根目录/
    ├── CMakeLists.txt     解析的构建脚本
    ├── main.cpp           源代码文件
    └── Dunkirk.jpg        加载的图像
# 二. 函数说明
## 1.CMakeLists.txt
    cmake_minimum_required(VERSION 3.1)  
    （必需：版本说明）
  
    project(imgdisplay) 
    （必需：定义项目名）
 
    find_package(OpenCV REQUIRED)
    （查找并加载OpenCV包配置）
 

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
    （要求编译器必须支持指定的C++标准）
 

    add_executable(imgdisplay main.cpp)
    （必需：可执行文件）
 

    target_link_libraries(imgdisplay PRIVATE
     （将OpenCV库链接到可执行文件）



## 2.main.cpp
```cpp
//OpenCV库的核心头文件和C++标准输入输出流库
#include <opencv2/opencv.hpp>
#include <iostream>  

//声明使用cv和std命名空间
using namespace cv;
using namespace std;

int main() {
   
   //定义字符串变量，存储图像文件路径
    string image_path = "./Dunkirk.jpg";

    //调用imread()函数加载指定路径的图像
    Mat image = imread(image_path);

  
  //检查图像是否加载成功，若文件不存在报错并返回
    if (image.empty()) {
        cout << "错误：无法加载图像，请检查路径是否正确。" << endl;
        return -1;
    }

   //创建名为"Display Image"的GUI窗口（自适应图像尺寸）
     namedWindow("Display Image", WINDOW_AUTOSIZE);

    //窗口中加载图像
    imshow("Display Image", image);

    //（0为一直等待，1000则等待1000ms）
    int key = waitKey(0);

    
    //检测是否按下s键，用指定输出路径将当前图像保存
    if (key == 's') {  
    
        string output_path = "saved_image.jpg";
        imwrite(output_path, image);
        cout << "图像已保存为 " << output_path << endl;

    //键入非s键时提示图像未保存
    } else {  
        cout << "图像未保存。" << endl;
    }

    //关闭窗口
    destroyAllWindows();

    return 0;
}
```
# 三. Cmake构建
## 1. 构建系统文件
   **cmake .**
   解析当前目录下的 CMakeLists.txt 文件
   检测系统环境
   生成Makefile

   **make**
   执行实际编译和链接
   生成最终可执行文件（imgdisplay）