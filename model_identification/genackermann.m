%% genackermann
folderPath = fullfile(pwd,"customAckermann_msg");
packagePath = fullfile(folderPath,"ackermann_msgs");
mkdir(packagePath)
mkdir(packagePath,"msg")
messageDefinition = {'float32 steering_angle', ...
'float32 steering_angle_velocity',...
'float32 speed',...
'float32 acceleration',...
'float32 jerk'};
 fileID = fopen(fullfile(packagePath,'msg','AckermannDrive.msg'),'w');
 fprintf(fileID,'%s\n',messageDefinition{:});
 fclose(fileID);

messageDefinition = {'std_msgs/Header header', ...
'ackermann_msgs/AckermannDrive drive'};
 fileID = fopen(fullfile(packagePath,'msg','AckermannDriveStamped.msg'),'w');
 fprintf(fileID,'%s\n',messageDefinition{:});
 fclose(fileID);


ros2genmsg(folderPath)