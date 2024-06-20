%% ros2
%help ros2 help ros2publisher
ros2 topic list
matlabnode = ros2node('matlabnode')
%sub = ros2subscriber(matlabnode,'/hamster2/pose')
sub = ros2subscriber(matlabnode,'/robot0/pose')
sub.LatestMessage.pose.position.x % 0.3066
sub.LatestMessage.pose.position.y % -3.3361
sub.LatestMessage.pose.position.z % 0.1361
[sub.LatestMessage.pose.position.x sub.LatestMessage.pose.position.y sub.LatestMessage.pose.position.z]
%sub.LatestMessage.pose.orientation
test_quat = [sub.LatestMessage.pose.orientation.x sub.LatestMessage.pose.orientation.y sub.LatestMessage.pose.orientation.z sub.LatestMessage.pose.orientation.w];
quat2eul(test_quat)*(180/pi) % last entry is yaw; if you go above 90 degreee you have to check if it is still correct...
sub2 = ros2subscriber(matlabnode,'/hamster2/command') % Unrecognized message type ackermann_msgs/AckermannDriveStamped. Use ros2 msg list to see available types.
% oder  /hamster2/interlock std_msgs/msg/Bool data:\true\~
[pub_acker,msg_acker] = ros2publisher(matlabnode,'/hamster2/command','ackermann_msgs/AckermannDriveStamped')
%[pub_acker,msg_acker] = ros2publisher(matlabnode,'hamster2/command','ackermann_msgs/AckermannDrive')

[pub,msg] = ros2publisher(matlabnode,'/hamster2/twist_command','geometry_msgs/Twist')
%sub2 = ros2subscriber(matlabnode,'/hamster2/twist_command')
msg.linear.x = 1;
msg.angular.z = 3;
% to send somethign to the Hamster
for i =1:100
    pause(0.1)
send(pub,msg)
end


%% for ackermann
% %ros2genmsg(folderpath)
% %msg_acker = ros2message('ackermann_msgs/AckermannDriveStamped')
% sometimes the hamster seems to just stop working (in ubuntu@hamster2
% there comes an error msg.


msg_acker.drive.speed = single(0.5); % maybe change i message definition float32 to float 64..then it shoudl work without "single"
msg_acker.drive.steering_angle =single(20);% seems to be in deg%single(pi);% pi/8;
 for i = 1:10
     pause(0.10)
 send(pub_acker,msg_acker)
%% 
 end

% folderPath = fullfile(pwd,"customAckermann_msg");
% packagePath = fullfile(folderPath,"ackermann_msgs");
% mkdir(packagePath)
% mkdir(packagePath,"msg")
% messageDefinition = {'float32 steering_angle', ...
% 'float32 steering_angle_velocity',...
% 'float32 speed',...
% 'float32 acceleration',...
% 'float32 jerk'};
%  fileID = fopen(fullfile(packagePath,'msg','AckermannDrive.msg'),'w');
%  fprintf(fileID,'%s\n',messageDefinition{:});
%  fclose(fileID);
% 
% messageDefinition = {'std_msgs/Header header', ...
% 'ackermann_msgs/AckermannDrive drive'};
%  fileID = fopen(fullfile(packagePath,'msg','AckermannDriveStamped.msg'),'w');
%  fprintf(fileID,'%s\n',messageDefinition{:});
%  fclose(fileID);
% 
% 
% ros2genmsg(folderPath)
%mkdir(packagePath,"srv")
%   Example:
%
%      % Create a custom message package folder in a local directory.
%      folderPath = fullfile(pwd,"ros2CustomMessages");
%      packagePath = fullfile(folderPath,"simple_msgs");
%      mkdir(packagePath)
% 
%      % Create a folder msg inside the custom message package folder.
%      mkdir(packagePath,"msg")
% 
%      % Create a .msg file inside the msg folder.
%      messageDefinition = {'int64 num'};
% 
%      fileID = fopen(fullfile(packagePath,'msg','Num.msg'),'w');
%      fprintf(fileID,'%s\n',messageDefinition{:});
%      fclose(fileID);
% 
%      % Create a folder srv inside the custom message package folder.
%      mkdir(packagePath,"srv")
% 
%      % Create a .srv file inside the srv folder.
%      serviceDefinition = {'int64 a'
%                           'int64 b'
%                           '---'
%                           'int64 sum'};
% 
%      fileID = fopen(fullfile(packagePath,'srv','AddTwoInts.srv'),'w');
%      fprintf(fileID,'%s\n',serviceDefinition{:});
%      fclose(fileID);
%
%      % Create a folder action inside the custom message package folder.
%      mkdir(packagePath,"action")
% 
%      % Create an .action file inside the action folder.
%      actionDefinition = {'int64 goal'
%                          '---'
%                          'int64 result'
%                          '---'
%                          'int64 feedback'};
% 
%      fileID = fopen(fullfile(packagePath,'action','Test.action'),'w');
%      fprintf(fileID,'%s\n',actionDefinition{:});
%      fclose(fileID);
%
%      % Generate custom messages from ROS 2 definitions in .msg, and .srv files.
%      ros2genmsg(folderPath)
%
%      % Generate custom messages and generate the zip file.
%      ros2genmsg(folderPath,CreateShareableFile=true)
%
%   See also ros2message, ros2.

%%


%msg = ros2message(msgType)

%Unrecognized message type ackermann_msgs/AckermannDriveStamped. Use ros2 msg list to see
%available types.

% or in terminal "The message type
% 'ackermann_msgs/msg/AckermannDriveStamped' is invalid

% https://wiki.ros.org/ackermann_msgs

% ackermann_msgs/AckermannDrive.msg
% 
% ## Driving command for a car-like vehicle using Ackermann steering.
% #  $Id$
% 
% # Assumes Ackermann front-wheel steering. The left and right front
% # wheels are generally at different angles. To simplify, the commanded
% # angle corresponds to the yaw of a virtual wheel located at the
% # center of the front axle, like on a tricycle.  Positive yaw is to
% # the left. (This is *not* the angle of the steering wheel inside the
% # passenger compartment.)
% #
% # Zero steering angle velocity means change the steering angle as
% # quickly as possible. Positive velocity indicates a desired absolute
% # rate of change either left or right. The controller tries not to
% # exceed this limit in either direction, but sometimes it might.
% #
% float32 steering_angle          # desired virtual angle (radians)
% float32 steering_angle_velocity # desired rate of change (radians/s)
% 
% # Drive at requested speed, acceleration and jerk (the 1st, 2nd and
% # 3rd derivatives of position). All are measured at the vehicle's
% # center of rotation, typically the center of the rear axle. The
% # controller tries not to exceed these limits in either direction, but
% # sometimes it might.
% #
% # Speed is the desired scalar magnitude of the velocity vector.
% # Direction is forward unless the sign is negative, indicating reverse.
% #
% # Zero acceleration means change speed as quickly as
% # possible. Positive acceleration indicates a desired absolute
% # magnitude; that includes deceleration.
% #
% # Zero jerk means change acceleration as quickly as possible. Positive
% # jerk indicates a desired absolute rate of acceleration change in
% # either direction (increasing or decreasing).
% #
% float32 speed                   # desired forward speed (m/s)
% float32 acceleration            # desired acceleration (m/s^2)
% float32 jerk                    # desired jerk (m/s^3)

