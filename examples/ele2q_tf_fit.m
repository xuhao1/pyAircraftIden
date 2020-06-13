freqres_ele2q = readfreqres('freqres_ele2q_5mps.csv');
ele2q_est = tfest(freqres_ele2q, 2);

s = tf('s');
%ele2q_est = (-4567.92*s+720.43)/(s*s*(s-32.26))*exp(-0.0748*s);
ele2q_esthover = (-2.735e09 *s^3 + 1.442e11* s^2 - 3.854e10*s + 4.184e12)/(  s^8 + 28.88*s^7 + 1.966e04*s^6 + 4.684e05*s^5 + 6.375e07*s^4 + 1.243e09*s^3 + 2.971e09*s^2 + 3.513e10*s + 2.537e10);

opts = bodeoptions('cstprefs');

opts.PhaseWrapping = 'on';

opts.PhaseWrappingBranch = -360;
figure(1)
clf()
title('ELE2Q FITl')
bode(ele2q_est, freqres_ele2q, ele2q_esthover, opts);
grid on;

figure(2)
clf()
title("ELE2QStep")
step(ele2q_est)
grid on;

%Cangvel = (4500*s^2+45000*s+40500)/(s^3+1800*s^2+810000*s);
Cangvel = pid(0.3, 0.05, 0.01);
%Cangvel = pid(0.05, 0.0, 0.0);
qloop = feedback(Cangvel*ele2q_est,1);
figure(3);
clf();
bode(qloop)
title('Qloop')
grid on;
figure(4)
clf()
title("Step Response of Pitchrate")
step(qloop)

Cang = pid(8, 0, 0);
%Cang = pid(2, 0, 0);

thetaloop = feedback(qloop*Cang/s, 1);
figure(5)
clf()
step(thetaloop)
grid on;
title("Step Response of Pitch Angle")
figure(6)
bode(thetaloop, opts)
grid on;
title('Transfer Function of Pitch Angle')
