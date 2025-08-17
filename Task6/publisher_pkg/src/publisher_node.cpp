#include <ros/ros.h>
#include "task6_pkg/trebleints.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "publisher_node");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<task6_pkg::trebleints>("trebleints", 10);

    while (pub.getNumSubscribers() == 0) {
        ros::Duration(0.1).sleep();
    }
       
        task6_pkg::trebleints msg;
        msg.a = 110;
        msg.b = 119;
        msg.c = 120;

        pub.publish(msg);       
        ros::Duration(0.5).sleep();
           
    return 0;
}
