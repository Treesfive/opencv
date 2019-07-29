# opencv
这里使用工具为Python3.5，OpenCV-Python包(cv2)，Numpy，matlablib

一段示例的代码：
```python
import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer


def test():
    import os
    im_file = 'demo/004545.jpg'
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    # im_file = '/media/longc/Data/data/2DMOT2015/test/ETH-Crossing/img1/000100.jpg'
    image = cv2.imread(im_file) # 读入图片

    model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'
    detector = FasterRCNN()
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.7)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show) # 写入图片
    cv2.imshow('demo', im2show)  # 显示图片
    cv2.waitKey(0)  # 等待键盘输入为ms级


if __name__ == '__main__':
    test()

```

## 图像读取
+ 图像读取:cv2.inread（文件名，标记）
+ 图像显示：cv2.inshow(显示窗口文件名，图片名称)
+ 图像读取：cv2.inwrite(保存的文件名，图片名称)


## OpenCV使用
### OpenCV-C++
+ **初始化图像对象，用Mat类**
> Mat M(3,2, CV_8UC3, Scalar(0,0,255));

第一行代码创建一个**行数（高度）为 3，列数（宽度）为 2 的图像，图像元素是 8 位无符号整数类型，且有三个通道**。图像的所有像素值被初始化为(0, 0,
255)。由于 **OpenCV 中默认的颜色顺序为 BGR**，因此这是一个全红色的图像。
+ **创建多通道图像,使用Mat类**
> Mat M(3,2, CV_8UC(5));//创建行数为 3，列数为 2，通道数为 5 的图像

+ **creat(）函数也创建图像对象**
> Mat M(2,2, CV_8UC3);//构造函数创建图像

> M.create(3,2, CV_8UC2);//释放内存重新创建图像

**creat(）函数无法设置图像像素的初始值**

+ **彩色，多维图像表示,多维矩阵**
```
Vec3b color; //用 color 变量描述一种 RGB 颜色
color[0]=255; //B 分量
color[1]=0; //G 分量
color[2]=0; //R 分量
```

+ **像素值的读写,at()函数,需要注意的是，如果要遍历图像，并不推荐使用 at()函数。使用这个函数的优点是代码的可读性高，但是效率并不是很高**

```
uchar value = grayim.at<uchar>(i,j);//读出第 i 行第 j 列像素值
grayim.at<uchar>(i,j)=128; //将第 i 行第 j 列像素值设置为 128
```

>如果要对图像进行遍历，可以参考下面的例程。这个例程创建了两个图像，分别是单通道的 grayim 以及 3 个通道的 colorim，然后对两个图像的所有像素值
进行赋值，最后现实结果。以下为代码：
```
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
Mat grayim(600, 800, CV_8UC1);
Mat colorim(600, 800, CV_8UC3);
//遍历所有像素，并设置像素值
for( int i = 0; i < grayim.rows; ++i)
for( int j = 0; j < grayim.cols; ++j )
grayim.at<uchar>(i,j) = (i+j)%255;
//遍历所有像素，并设置像素值
for( int i = 0; i < colorim.rows; ++i)
for( int j = 0; j < colorim.cols; ++j )
{
Vec3b pixel;
pixel[0] = i%255; //Blue
pixel[1] = j%255; //Green
pixel[2] = 0; //Red
colorim.at<Vec3b>(i,j) = pixel;
}
//显示结果
imshow("grayim", grayim);
imshow("colorim", colorim);
waitKey(0);
return 0;
}
```

+  **MatIterator 遍历像素**

```#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
Mat grayim(600, 800, CV_8UC1);
Mat colorim(600, 800, CV_8UC3);
//遍历所有像素，并设置像素值
MatIterator_<uchar> grayit, grayend;
for( grayit = grayim.begin<uchar>(), grayend =
grayim.end<uchar>(); grayit != grayend; ++grayit)
*grayit = rand()%255;
//遍历所有像素，并设置像素值
MatIterator_<Vec3b> colorit, colorend;
for( colorit = colorim.begin<Vec3b>(), colorend =
colorim.end<Vec3b>(); colorit != colorend; ++colorit)
{
(*colorit)[0] = rand()%255; //Blue
(*colorit)[1] = rand()%255; //Green
(*colorit)[2] = rand()%255; //Red
}
//显示结果
imshow("grayim", grayim);
imshow("colorim", colorim);
waitKey(0);
return 0;
}
```


+ **选取图像某行或某列用函数row()或col()**

+ **选区图像中的局部区域,Range 类还提供了一个静态方法 all()，这个方法的作用如同 Matlab 中的“:”，表示所有的行或者所有的列**

```
//创建一个单位阵
Mat A = Mat::eye(10, 10, CV_32S);
//提取第 1 到 3 列（不包括 3）
Mat B = A(Range::all(), Range(1, 3));
//提取 B 的第 5 至 9 行（不包括 9）
//其实等价于 C = A(Range(5, 9), Range(1, 3))
Mat C = B(Range(5, 9), Range::all());
```

+ **提取图像的感兴趣区域，用rect()函数，或者Range(）对象来选择感兴趣区域**

```
//创建宽度为 320，高度为 240 的 3 通道图像
Mat img(Size(320,240),CV_8UC3);
//roi 是表示 img 中 Rect(10,10,100,100)区域的对象
Mat roi(img, Rect(10,10,100,100));
除了使用构造函数，还可以使用括号运算符， 如下：
Mat roi2 = img(Rect(10,10,100,100));
当然也可以使用 Range 对象来定义感兴趣区域，如下：
//使用括号运算符
Mat roi3 = img(Range(10,100),Range(10,100));
```

+ **读图像文件**

` Mat imread(const string& filename, int flags=1 )`

> 很明显参数 filename 是被读取或者保存的图像文件名；在 imread()函数中，flag 参数值有三种情况：flag>0，该函数返回 3 通道图像，如果磁盘上的图像文件是单通道的灰度图像，则会被强制转为 3 通道；flag=0，该函数返回单通道图像，如果磁盘的图像文件是多通道图像，则会被强制转为单通道；flag<0，则函数不对图像进行通道转换


+ **写图像文件**
```
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char* argv[])
{
//读入图像，并将之转为单通道图像
Mat im = imread("lena.jpg", 0);
//请一定检查是否成功读图
if( im.empty() )
{
cout << "Can not load image." << endl;
return -1;
}
//进行 Canny 操作，并将结果存于 result
Mat result;
Canny(im, result, 50, 150);
//保存结果
imwrite("lena-canny.png", result);
return 0;
}
```

+ **读视频文件VideoCapture()**

```#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
//打开第一个摄像头
//VideoCapture cap(0);
//打开视频文件
VideoCapture cap("video.short.raw.avi");
//检查是否成功打开
if(!cap.isOpened())
{
cerr << "Can not open a camera or file." << endl;
return -1;
}
Mat edges;
//创建窗口
namedWindow("edges",1);
for(;;)
{
Mat frame;
//从 cap 中读一帧，存到 frame
cap >> frame;
//如果未读到图像
if(frame.empty())
break;
//将读到的图像转为灰度图
cvtColor(frame, edges, CV_BGR2GRAY);
//进行边缘提取操作
Canny(edges, edges, 0, 30, 3);
//显示结果
imshow("edges", edges);
//等待 30 秒，如果按键则推出循环
if(waitKey(30) >= 0)
break;
}
//退出时会自动释放 cap 中占用资源
return 0;
}
```

+ **写视频文件VideoWritter()**

> 使用 OpenCV 创建视频也非常简单，与读视频不同的是，你需要在创建视频时设置一系列参数，包括：文件名，编解码器，帧率，宽度和高度等。编解码器
使用四个字符表示，可以是 CV_FOURCC('M','J','P','G')、 CV_FOURCC('X','V','I','D')及CV_FOURCC('D','I','V','X')等。如果使用某种编解码器无法创建视频文件，请尝试其他的编解码器。将图像写入视频可以使用 VideoWriter::write()函数， VideoWriter 类中也重载了<<操作符，使用起来非常方便。另外需要注意：待写入的图像尺寸必须与创建视频时指定的尺寸一致。下面例程演示了如何写视频文件。本例程将生成一个视频文件，视频的第 0帧上是一个红色的“0”，第 1 帧上是个红色的“1”，以此类推，共 100 帧。

```
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
//定义视频的宽度和高度
Size s(320, 240);
//创建 writer，并指定 FOURCC 及 FPS 等参数
VideoWriter writer = VideoWriter("myvideo.avi",
CV_FOURCC('M','J','P','G'), 25, s);
//检查是否成功创建
if(!writer.isOpened())
{
cerr << "Can not create video file.\n" << endl;
return -1;
}
//视频帧
Mat frame(s, CV_8UC3);
for(int i = 0; i < 100; i++)
{
//将图像置为黑色
frame = Scalar::all(0);
//将整数 i 转为 i 字符串类型
char text[128];
snprintf(text, sizeof(text), "%d", i);
//将数字绘到画面上
putText(frame, text, Point(s.width/3, s.height/3),
FONT_HERSHEY_SCRIPT_SIMPLEX, 3,
Scalar(0,0,255), 3, 8);
//将图像写入视频
writer << frame;
}
//退出程序时会自动关闭视频文件
return 0;
}
```
