#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <typeinfo>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <cstring>
#include <csignal>

#include <unordered_map>
using namespace std;

int serverSocket=-1;
int clientSocket=-1;
int port=1235;
void connect_server(){
    

    
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        cerr<<"socket creation error "<<strerror(errno) <<endl;
    }
    int enable=1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
        cerr<<"stsockopt(SO_REUSEADDR) failed"<<endl;


  // Set up server address
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr =  htonl(INADDR_ANY); // Replace with desired port
    serverAddress.sin_port = htons(port);

  // Bind the socket
    if (bind(serverSocket, (sockaddr*)&serverAddress, sizeof(serverAddress)) == -1) {
    // Handle error

        cout<<"binding error:"<< strerror(errno) <<endl;
        
   
    }
  

  // Listen for connections
    if (listen(serverSocket, 1) == -1) {
            cout<<"listen error "<<strerror(errno) <<endl;

        // Handle error
    }

    std::cout << "Server listening on port ..." << std::endl;

  // Accept a connection
    sockaddr_in clientAddress;
    socklen_t clientAddressSize = sizeof(clientAddress);
    while ((clientSocket = accept(serverSocket, (sockaddr*)&clientAddress, &clientAddressSize)) == -1) {
        std::cerr << "Client socket error: " << strerror(errno) << std::endl;
    }

    std::cout << "Client connected from " << inet_ntoa(clientAddress.sin_addr) << ":" << ntohs(clientAddress.sin_port) << std::endl;

  

    sleep(2);
    while (true){
        char buffer[1024];
        memset(buffer, 0, 1024);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);

        struct timeval timeout;
        timeout.tv_sec = 0.5;
        timeout.tv_usec = 0;

        int activity = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0) {
            std::cerr << "Select error: " << strerror(errno) << std::endl;
        } else if (activity == 0) {
            std::cout << "Timeout: connect No data received for " << 0.5 << " seconds." << std::endl;
        }
        else{
            ssize_t bytesReceived = recv(clientSocket, buffer, 1024 - 1, 0);
            if (bytesReceived < 0) {
                std::cerr << "Error receiving message: " << strerror(errno) << std::endl;
            } else if (bytesReceived == 0) {
                std::cout << "Client disconnected." << std::endl;
            }
            std::cout << "Received echo: " << buffer << std::endl;

            if(strncmp(buffer,"Connection established from client.",1024)==0){
                break;
            }
   
        }
    }


}
//qt_pose signal to pose_cpp sigterm
//pose_cpp signal to decision_python


void connect_to_image_server(){
    
}
struct MoveInfo {
    int repetitions;
    std::string lastState;
};

void handle_sigint(int sig) {
    std::cout << "\nCaught signal " << sig << " . Exiting gracefully...\n";

    //send to pose_cpp signal for terminate
    
    close(clientSocket); // Close the client socket
    close(serverSocket); // Close the server socket


    exit(0); // Exit the program
}




int main(){
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    char move_name[50];
    float probability;
    char status[50];
    int repetition;
    std::unordered_map<std::string, MoveInfo> moves;
    //create python  sovket
    connect_server();



    //run pose cpp program and decison python program

     
    while(true){
        char buffer[1024];
        memset(buffer, 0, 1024);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);
        float timeout_sec=0.2;
        struct timeval timeout;
        timeout.tv_sec =timeout_sec;
        timeout.tv_usec = timeout_sec*1000000;

        int activity = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0) {
            std::cerr << "Select error: " << strerror(errno) << std::endl;
        } 
        else if (activity == 0) {
            std::cout << "Timeout: No data received for " << 0.5 << " seconds." << std::endl;
        }
        else{
            ssize_t bytesReceived = recv(clientSocket, buffer, 1024 - 1, 0);
            if (bytesReceived < 0) {
                std::cerr << "Error receiving message: " << strerror(errno) << std::endl;
            } 
            else if (bytesReceived == 0) {
                std::cout << "Echo can not received." << std::endl;
            }
            else{
                //where to process message and do all work
                
                std::cout << "Received echo: " << buffer << std::endl;

               

                int result = sscanf(buffer, "%s %f %s %d", move_name, &probability, status, &repetition);
                auto it = moves.find(move_name);
                if (it != moves.end()) {
                    // Move found, access its information
                    std::cout << "Move: " << move_name << std::endl;
                    std::cout << "Repetitions: " << it->second.repetitions << std::endl;
                    std::cout << "Last State: " << it->second.lastState << std::endl;

                    // Modify the move information
                    it->second.repetitions = repetition;
                    it->second.lastState = status;

                    std::cout << "Updated Repetitions: " << it->second.repetitions << std::endl;
                    std::cout << "Updated Last State: " << it->second.lastState << std::endl;
                } else {
                    // Move not found
                    std::cout << "Move not found!" << std::endl;
                    
                    MoveInfo newMoveInfo = {repetition, status};
                    // Insert the new move into the map
                    moves[move_name] = newMoveInfo;
                    //check if new move added
                    auto newIt = moves.find(move_name);
                    if (newIt != moves.end()) 
                    {
                        
                    } 
                    else 
                    {
                        std::cerr << "Failed to add new move!" << std::endl;
                    }
                
                }




                

            }

            

             
   
   
        }
        }


}