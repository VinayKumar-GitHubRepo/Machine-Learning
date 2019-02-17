function [ pos1Fin pos2Fin pos3Fin pos4Fin ] = InvKin( coord )
% Apply Inverse Kinematics equations to compute the joints angles given a
% desired position for the end effector

% Robotic Arm dimensions [cm]
a1 = 11.5;
a2 = 11.5;
a3 = 12;

% Inverse kinematic parameters
phi = 0;
px = coord(1) -a3*cosd(phi);
py = coord(2) -a3*sind(phi);

% Compute pos1
pos1Fin = round(coord(3));

% Compute pos3
c2 = (px^2 +py^2 -a1^2 -a2^2)/(2*a1*a2);
s2 =- sqrt(1-c2^2);
pos3 = rad2deg(atan2(s2, c2));

% Compute pos 2
c2 = (px^2 +py^2 -a1^2 -a2^2)/(2*a1*a2);
s2 =- sqrt(1-c2^2);
s1 = ((a1 +a2*c2)*py -a2*s2*px)/(px^2 +py^2);
c1 = ((a1 +a2*c2)*px +a2*s2*py)/(px^2 +py^2);
pos2 = rad2deg(atan2(s1, c1));

% Compute pos 4
pos4 = phi -pos3 -pos2;

% Return angles
pos2Fin = pos2;
pos3Fin = pos3;
pos4Fin = pos4;

end

