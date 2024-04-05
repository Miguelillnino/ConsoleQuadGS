
#include <cmath>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <numeric>
#include <vector>

/*
*******************************************************
******************************************************
*  Version of quadrilateral-detection software 1.0.0" 
*******************************************************
*******************************************************
* Sequencial program that detects quadrilateral polygons
* Fuctions:
*   angleCos()
*   findQuads()
*   main()
********************************************************
******************************************************
*******************************************************
******************************************************
*/



/**
    * Calculates the cosine of the angle between Points from Vector
    * The cosine of the angle between two vectors can be calculated
    * using the dot product and the magnitudes of the vectors.
    *
    * @param values Container three 2d-Points `p0`, `p1` and , `p2`.
    * @return the double of `result`.
    */


double angleCos(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / std::sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}


/**
    * Returns a sequence of quadrilaterals detected on the image.
    * Metholody:
    *   -Canny
    *   -Dilation
    *   -To Find Countours
    *   -Aprroximation of Poligonal Curves
    *   -Poligon 4 Vertices, Convex and Area greater than 1000
    *   -Calculate maximum cosine between angles
    *   -All angles are ~90 degree
    *
    * @param values Container a cv::Mat image(Gray-Scale) and Vector<Vector<Point>> quads
    * @return the double of `result`.
    */

static void findQuads(const cv::Mat& image, std::vector<std::vector<cv::Point> >& quads)
{
    quads.clear();
    
    cv::Mat edges;
    cv::Mat dilate;

    for (double thrs = 0; thrs < 255; thrs += 26)
    {
        //Detect Edges in Gray Image
        cv::Canny(image, edges, thrs, 255.0, 5);
        //Defined the kernel of dilation
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        //Dilate the Gray image
        int iterations = 1; // Number of iterations
        cv::dilate(edges, dilate, kernel, cv::Point(-1, -1), iterations);

        std::vector<std::vector<cv::Point>> contornos;

        if (!dilate.empty()) {
            cv::findContours(dilate, contornos, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        }
        std::vector<cv::Point> approx;

        // Test each contour
        for (size_t i = 0; i < contornos.size(); i++)
        {
            // approximate contour with accuracy proportional
            // to the contour perimeter

            cv::approxPolyDP(contornos[i], approx, cv::arcLength(contornos[i], true) * 0.02, true);

            // quads contours should have 4 vertices
            // relatively large area (to filter out noisy contours)
            // and be convex.
            // area may be positive or negative - in accordance with the
            // contour orientation
            if (approx.size() == 4 &&
                fabs(cv::contourArea(approx)) > 1000 && cv::isContourConvex(approx))
            {
                double maxCosine = 0;
                for (int j = 2; j < 5; j++)
                {
                    // find the maximum cosine of the angle between joint edges
                    double cosine = fabs(angleCos(approx[j % 4], approx[j - 2], approx[j - 1]));
                    maxCosine = MAX(maxCosine, cosine);
                }

                // if cosines of all angles are small
                // (all angles are ~90 degree) then save a quad
                if (maxCosine < 0.3) {
                    quads.push_back(approx);
                }
            }
        }
    }

 }


/*
*Testing quads detction using the camera by default.
* 
*/

int main()
{
    std::cout << "Version of quadrilateral-detection software 1.0.0" << std::endl;

    int deviceID = 0; // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API

    //Initialize Camera and reading the frames from the camera
    cv::VideoCapture cap_video;
    cap_video.open(deviceID, apiID);

    if (!cap_video.isOpened()) {
        std::cout << "Error reading video file" << std::endl;
        return -1;
    }

    //Set up the camera 
    
    //std::cout << cap.get(cv::CV_CAP_PROP_FPS) << std::endl;
    std::cout << cap_video.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    cap_video.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    cap_video.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_video.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    double Width = cap_video.get(cv::CAP_PROP_FRAME_WIDTH);
    double Height = cap_video.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Width= "<< Width << std::endl;
    std::cout << "Height= " << Height << std::endl;

    std::cout << "Start grabbing" << std::endl
        << "Press any ESC to terminate" << std::endl;
    
   /*
    * Code to read Frame by Frame
    */

    cv::Mat frame; 
    for (;;)
    {
        cap_video.read(frame);
        if (frame.empty())
            break;
        
        //Convert image in gray scale
        cv::Mat imgGray;
        cv::cvtColor(frame,imgGray, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point>> quads;
        
        findQuads(imgGray, quads);
   
        //Print detected quads
        for (const auto& quad : quads) {
            std::cout << "Quad:" << std::endl;
            for (const auto& point : quad) {
                std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
            }
            std::cout << std::endl;
        }
        //draw quads in image from frame
        cv::polylines(imgGray, quads, true, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        cv::imshow("Imagen", imgGray);
        
        // Press ESC for exit
        char c = (char)cv::waitKey(25);
        if (cv::waitKey(30) == 27) { 
            break;
        }
        
    }
}
