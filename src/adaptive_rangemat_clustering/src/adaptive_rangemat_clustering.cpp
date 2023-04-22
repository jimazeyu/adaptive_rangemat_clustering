#include "utility.h"
#include <visualization_msgs/MarkerArray.h>

class RangematClustering{
private:

    ros::NodeHandle nh;

    // 第几帧，用于测试计算速度以及准确率
    int frameCount=0;

    // 搜索范围内的邻域点
    const int neighborRangeSize = 3; 

    // 聚类距离系数
    const float clusterDistance = 1.2;

    // 聚类最小点数
    const int minClusterSize = 5;

    // 订阅点云
    ros::Subscriber subLaserCloud;

    // 发布地面点、分割后的点云、分割后的点云的包围盒
    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubClusterBox;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;

    PointType nanPoint; // 填充无效点

    cv::Mat rangeMat; // 距离矩阵
    cv::Mat labelMat; // 标签矩阵
    cv::Mat groundMat; // 地面点标签矩阵
    int labelCount;

    std_msgs::Header cloudHeader;

    // 用于存储邻域点的坐标
    std::vector<std::pair<int8_t, int8_t> > neighborIterator; 

    uint16_t *allPushedIndX; 
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; 
    uint16_t *queueIndY;

public:
    RangematClustering():
        nh("~"){

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &RangematClustering::cloudHandler, this);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        
        pubClusterBox = nh.advertise<visualization_msgs::MarkerArray>("/cluster_box", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    // pcl点云手动需要初始化
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        // 搜索范围内的邻域点
        std::pair<int8_t, int8_t> neighbor;
        for (int i=-neighborRangeSize; i<=neighborRangeSize; i++){
            for (int j=-neighborRangeSize; j<=neighborRangeSize; j++){
                if (abs(i) > 0 || abs(j) > 0){
                    neighbor.first = i; neighbor.second = j; neighborIterator.push_back(neighbor);
                }
            }
        }

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    // 参数初始化
    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloudPure->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    }

    ~RangematClustering(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // 移除无效点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        
        // 用chrono记录开始时间(单位：ms)
        auto start = std::chrono::steady_clock::now();
        
        // 1. 将点云转换为pcl格式
        copyPointCloud(laserCloudMsg);
        // 2. 将扫帧投射到rangeMat上
        projectPointCloud();
        // 3. 移除地面点
        groundRemoval();
        // 4. 点云分割
        cloudSegmentation();
        // 6. 发布点云及分割结构
        publishResult();
        // 7. 参数重置
        resetParameters();

        // 计算时间
        auto end = std::chrono::steady_clock::now();
        std::ofstream outfile;
        outfile.open("/home/jimazeyu/桌面/ros_ws/adaptive_rangemat_clustering/results/kitti_clustering_time.txt", std::ios::app);        
        outfile << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << std::endl;
        outfile.close();
        
        // 用于测试计算速度以及准确率
        frameCount++;
    }

    // 将点云投影到rangeMat上
    void projectPointCloud(){
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat(1代表地面，0代表非地面，-1代表无效点)
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // 将地面点和无效点标记为-1，后续聚类时不考虑
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // 点云分割
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        // 用于发布分割后的点云
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    // BFS搜索聚类
    void labelComponents(int row, int col){
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                // 计算两点之间角度
                float alphaX = segmentAlphaX*abs((*iter).second);
                float alphaY = segmentAlphaY*abs((*iter).first);
                alpha = sqrt(alphaX*alphaX + alphaY*alphaY);

                d1 = rangeMat.at<float>(fromIndX, fromIndY);
                d2 = rangeMat.at<float>(thisIndX, thisIndY);

                if(fabs(d1-d2)<clusterDistance*d1*alpha){

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // 给足够大的聚类标记
        if (allPushedIndSize >= minClusterSize)++labelCount;
        else{
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishResult(){
        sensor_msgs::PointCloud2 laserCloudTemp;
        // 发布地面点点云
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // 发布去除地面点后分割出来的点云
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // 发布每个cluster的包围盒
        if (pubClusterBox.getNumSubscribers() != 0){
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr > > clusters;
            clusters.resize(labelCount);
            for (size_t i = 0; i < labelCount; ++i){
                pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>());
                clusters[i] = tmp;
            }
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        pcl::PointXYZI tmpPoint;
                        tmpPoint.x = fullCloud->points[j + i*Horizon_SCAN].x;
                        tmpPoint.y = fullCloud->points[j + i*Horizon_SCAN].y;
                        tmpPoint.z = fullCloud->points[j + i*Horizon_SCAN].z;
                        tmpPoint.intensity = fullCloud->points[j + i*Horizon_SCAN].intensity;
                        clusters[labelMat.at<int>(i,j)-1]->push_back(tmpPoint);
                    }
                }
            }
            visualization_msgs::MarkerArray marker_array;
            for (int i = 0; i < clusters.size()-1; i++)
            {
                Eigen::Vector4f min, max;
                pcl::getMinMax3D(*clusters[i], min, max);

                visualization_msgs::Marker marker;
                marker.header = cloudHeader;
                marker.header.frame_id = "base_link";
                marker.ns = "adaptive_rangemat_clustering";
                marker.id = i;
                marker.type = visualization_msgs::Marker::LINE_LIST;

                geometry_msgs::Point p[24];
                {
                    p[0].x = max[0];
                    p[0].y = max[1];
                    p[0].z = max[2];
                    p[1].x = min[0];
                    p[1].y = max[1];
                    p[1].z = max[2];
                    p[2].x = max[0];
                    p[2].y = max[1];
                    p[2].z = max[2];
                    p[3].x = max[0];
                    p[3].y = min[1];
                    p[3].z = max[2];
                    p[4].x = max[0];
                    p[4].y = max[1];
                    p[4].z = max[2];
                    p[5].x = max[0];
                    p[5].y = max[1];
                    p[5].z = min[2];
                    p[6].x = min[0];
                    p[6].y = min[1];
                    p[6].z = min[2];
                    p[7].x = max[0];
                    p[7].y = min[1];
                    p[7].z = min[2];
                    p[8].x = min[0];
                    p[8].y = min[1];
                    p[8].z = min[2];
                    p[9].x = min[0];
                    p[9].y = max[1];
                    p[9].z = min[2];
                    p[10].x = min[0];
                    p[10].y = min[1];
                    p[10].z = min[2];
                    p[11].x = min[0];
                    p[11].y = min[1];
                    p[11].z = max[2];
                    p[12].x = min[0];
                    p[12].y = max[1];
                    p[12].z = max[2];
                    p[13].x = min[0];
                    p[13].y = max[1];
                    p[13].z = min[2];
                    p[14].x = min[0];
                    p[14].y = max[1];
                    p[14].z = max[2];
                    p[15].x = min[0];
                    p[15].y = min[1];
                    p[15].z = max[2];
                    p[16].x = max[0];
                    p[16].y = min[1];
                    p[16].z = max[2];
                    p[17].x = max[0];
                    p[17].y = min[1];
                    p[17].z = min[2];
                    p[18].x = max[0];
                    p[18].y = min[1];
                    p[18].z = max[2];
                    p[19].x = min[0];
                    p[19].y = min[1];
                    p[19].z = max[2];
                    p[20].x = max[0];
                    p[20].y = max[1];
                    p[20].z = min[2];
                    p[21].x = min[0];
                    p[21].y = max[1];
                    p[21].z = min[2];
                    p[22].x = max[0];
                    p[22].y = max[1];
                    p[22].z = min[2];
                    p[23].x = max[0];
                    p[23].y = min[1];
                    p[23].z = min[2];
                }
                for (int i = 0; i < 24; i++)
                {
                    marker.points.push_back(p[i]);
                }

                // 以单位矩阵表示姿态
                marker.pose.orientation.w = 1.0;

                marker.scale.x = 0.05;
                marker.color.a = 1.0;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.5;
                marker.lifetime = ros::Duration(10);
                marker_array.markers.push_back(marker);

                // 记录数据到文件中（清空原文件,然后一行行读入）
                std::ofstream outfile;
                outfile.open("/home/jimazeyu/桌面/ros_ws/adaptive_rangemat_clustering/results/kitti_clustering.txt", std::ios::app);
                outfile << frameCount << " " << min[0] << " " << min[1] << " " << min[2] << " " << max[0] << " " << max[1] << " " << max[2] << std::endl;
                outfile.close();

            }

            if (marker_array.markers.size())
            {
                pubClusterBox.publish(marker_array);
            }
        }
    }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "adaptive_rangemat_clustering");
    RangematClustering RC;
    ROS_INFO("\033[1;32m---->\033[0m Adaptive RangeMat Clustering Started.");
    ros::spin();
    return 0;
}
