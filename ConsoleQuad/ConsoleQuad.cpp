
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

    /**
    * Convert an Int Vector to Float Vector.
    *
    * @param values Container a vector `intVector`.
    * @return the vector of `floatVector`.
    */

/*
std::vector<double> castToIntVectorToDouble(const std::vector<int>& intVector) {
    std::vector<double> DoubleVector(intVector.size());

    // Iterate over each element and cast to float
    for (size_t i = 0; i < intVector.size(); ++i) {
        DoubleVector[i] = static_cast<float>(intVector[i]);
    }

    return DoubleVector;
}
*/



/**
    * substract two Vectors.
    *
    * @param values Container two vectors `v1` adn `v2`.
    * @return the double vector of `result`.
    */

std::vector<double> subtractDoubleVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    // Both vectors have the same dimension
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors must have the same size for subtraction." << std::endl;
        return std::vector<double>(); // Return an empty vector
    }

    // Create a vector to store the result
    std::vector<double> result;
    result.reserve(v1.size()); // Reserve memory for efficiency

    // Subtract corresponding elements
    for (size_t i = 0; i < v1.size(); ++i) {
        result.push_back(v1[i] - v2[i]);
    }

    return result;
}

/**
    * Calculates the cosine of the angle between two vectors
    * The cosine of the angle between two vectors can be calculated
    * using the dot product and the magnitudes of the vectors.
    *
    * @param values Container three vectors `p0`, `p1` and , `p2`.
    * @return the double vector of `result`.
    */


double angle_cos(const std::vector<double>& p0, const std::vector<double>& p1, const std::vector<double>& p2){
    double angleCosResult = -3.14;
    std::vector<double> d1 = subtractDoubleVectors(p0, p1);
    std::vector<double> d2 = subtractDoubleVectors(p2, p1);
    
    //COROBORAR QUE EL PRODUCTO INTERNO SE DEVUELVA COMO vector o double
    double aData = std::inner_product(d1.begin(), d1.end(), d2.begin(), 0.0);
    double bData = std::inner_product(d1.begin(), d1.end(), d1.begin(), 0.0);
    double cData = std::inner_product(d2.begin(), d2.end(), d2.begin(), 0.0);

    angleCosResult = fabs(aData / sqrt(bData*cData) );

    return angleCosResult;
}






int main()
{
    std::cout << "Version of quadrilateral-detection software 1.0.0" << std::endl;

    // Get the current time before starting the operation
    auto start = std::chrono::high_resolution_clock::now();
    // Get the current time after finishing the operation
    auto end = std::chrono::high_resolution_clock::now();
    
    //Open a Video Cam from one device
    cv::VideoCapture cap_video;
    cap_video.open(0);

    if (cap_video.isOpened() == false) {
        std::cout << "Error reading video file" << std::endl;
    }

    //std::cout << cap.get(cv::CV_CAP_PROP_FPS) << std::endl;
    std::cout << cap_video.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    cap_video.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('U', 'Y', 'V', 'Y'));
    
    cap_video.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_video.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    double Width = cap_video.get(cv::CAP_PROP_FRAME_WIDTH);
    double Height = cap_video.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Width= "<< Width << std::endl;
    std::cout << "Height= " << Height << std::endl;

    auto cap_duration = std::chrono::seconds(10);
    auto current = std::chrono::high_resolution_clock::now();

    while((current - start) < cap_duration){

        cv::Mat frame;
        cap_video >> frame;

        if (frame.empty())
            break;

        cv::imshow("Frame", frame);

        char c = (char) cv::waitKey(25);

        if (c == 27)
            break;
        }
        


    //std::cout << cv::getBuildInformation() << std::endl;

    
    
    
    
    // Calculate the duration of the operation
    std::chrono::duration<double> duration = end - start;
    // Print the duration in seconds
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;


    
}

//This code is to calculate the max_cos
/*
double max_cos = -1; // Initialize with a value less than -1
for (int i = 0; i < 4; ++i) {
    double cos_value = angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]);
    max_cos = std::max(max_cos, cos_value);
}
*/
