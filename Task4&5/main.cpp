
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
   
    string image_path = "./Dunkirk.jpg";
    Mat image = imread(image_path);

  
    if (image.empty()) {
        cout << "错误：无法加载图像，请检查路径是否正确。" << endl;
        return -1;
    }

   
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    
    int key = waitKey(0);

    
    if (key == 's') {  
    
        string output_path = "saved_image.jpg";
        imwrite(output_path, image);
        cout << "图像已保存为 " << output_path << endl;
    } else {  
        cout << "图像未保存。" << endl;
    }

    
    destroyAllWindows();

    return 0;
}
