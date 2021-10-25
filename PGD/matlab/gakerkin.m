clc; close all; clear all

A = [2, 0, 0; 0, 2, 1; 0, 1, 2];
x = [1, 0, 0]'; b = [2, 0, 0]';

V = [0, 1, 0; 0, 0, 1]'; 

gleft = V' * A * V;
gright = V'*b;
gy = gleft \gright;