#include <iostream>
#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <cstring>

int imageServerSocket;
int imageClientSocket;
int imageServerSocketPort = 8080; // Example port for image server

cv::Mat receiveFrame(int clientSocket) {
    int bufferSize;
    

    std::vector<uchar> buffer(bufferSize);
   int bytesRead = recv(clientSocket, buffer.data(), bufferSize, 0);
    if (bytesRead <= 0) {
        std::cerr << "Failed to read buffer data: " << strerror(errno) << std::endl;
        return cv::Mat();
    }

    cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Failed to decode frame" << std::endl;
    }
    return frame;
}

void processFrames() {
    while (true) {
        cv::Mat frame = receiveFrame(imageClientSocket);
        if (frame.empty()) {
            std::cerr << "Error receiving frame or client disconnected" << std::endl;
            break;
        }
        cv::imshow("Received Frame", frame);
        if (cv::waitKey(30) >= 0) break; // Adjust to control the frame rate
    }

    close(imageClientSocket);
    close(imageServerSocket);
}

void connect_image_server() {
    
      int sock = 0, valread;
   struct sockaddr_in serv_addr;
   char *hello = "Hello from client";
   char buffer[1024] = {0};
   if ((imageClientSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
       printf("\n Socket creation error \n");
   }
   serv_addr.sin_family = AF_INET;
   serv_addr.sin_port = htons(8080);
   // Convert IPv4 and IPv6 addresses from text to binary form
   if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) {
       printf("\nInvalid address/ Address not supported \n");
   }
   if (connect(imageClientSocket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
       printf("\nConnection Failed \n");
   }
    processFrames();
}

int main() {
    connect_image_server();
    return 0;
}
