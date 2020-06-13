function [sys] = readfreqres(name)
%READFREQRES 此处显示有关此函数的摘要
%   此处显示详细说明
d = csvread(name);
freq = d(:,1);
resp = d(:,2) + d(:,3)*1i;
sys = frd(resp, freq);
end

