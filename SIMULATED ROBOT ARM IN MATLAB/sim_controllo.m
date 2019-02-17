%% 
clear all
close all
clc

% warning('off');

%% SETUP
% Arudino variables
a = arduino('COM3', 'Uno');
x1pin = 'A0';
y1pin = 'A1';
x2pin = 'A2';
y2pin = 'A3';
x_sens = 5;
y_sens = 5;
z_sens = .5;
grip_sens = .5;

% Set starting pos
xCoord = 20;
yCoord = 16;
zAngle = 90;
gripAngle = 0.15;

%% ROBOTICS TOOLBOX
% Declare links
L(1) = Link([0 0 0 pi/2 0]);
L(2) = Link([0 0 11.5 0 0]);
L(3) = Link([0 0 11.5 0 0]);
L(4) = Link([0 0 12 0 0]);
% Create serial kinematic chain
four_link = SerialLink(L, 'name', 'four link');


%% SIMULATION

while(1)
   
    % Read joysticks
    zVal = readVoltage(a, x1pin) -2.5; 
    xVal = readVoltage(a, y1pin) -2.5;
    yVal = readVoltage(a, x2pin) -2.5;
    gripVal = readVoltage(a, y2pin) -2.5;
    
    % Print for debug
    % xVal, yVal, zVal, gripVal
    
    % Compute increment
    if abs(zVal) > 1
        zIncr = zVal/z_sens;
    else
        zIncr = 0;
    end
    if abs(xVal) > 1
        xIncr = xVal/x_sens;
    else
        xIncr = 0;
    end
    if abs(yVal) > 1
        yIncr = yVal/y_sens;
    else 
        yIncr = 0;
    end
    if abs(gripVal) > 1
        gripIncr = gripVal/grip_sens;
    else
        gripIncr = 0;
    end
    
    % Update position
    xCoord = xCoord + xIncr;
    yCoord = yCoord + yIncr;
    zAngle = zAngle + zIncr;
    gripAngle = gripAngle + gripIncr;
    
    % Use robotics toolbox to find angles
    T = transl([xCoord*cosd(zAngle) xCoord*sind(zAngle) yCoord]);
    q = four_link.ikine(T, [0 0 0 0], [1 1 1 0 0 0 ], 'u');
    
    % ALTERNATIVE TO THE TWO PREVIOUS LINES: Use inverse kinematics equations to find angles
    %[pos1Fin, pos2Fin, pos3Fin, pos4Fin] = InvKin([xCoord yCoord zAngle]);
    %q = [deg2rad(pos1Fin), deg2rad(pos2Fin), deg2rad(pos3Fin), deg2rad(pos4Fin)];
    
    % Show robotic arm
    figure(1);
    four_link.plot(q), grid on;
    
    pause(0.05);
    
end
