#include <ros/ros.h>
#include "task6_pkg/trebleints.h"

void callback(const task6_pkg::trebleints::ConstPtr& msg) {
    ROS_INFO(" %ld %ld %ld ", 
             msg->a, msg->b, msg->c);            
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "publisher_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("trebleints", 10, callback);
    ros::spin();
    return 0;
}
