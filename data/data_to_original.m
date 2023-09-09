mpc=runpf('case14');
P = -mpc.bus(:, 3);
len_P = length(P);
for i=1:length(mpc.gen(:, 1))
    a = mpc.gen(i, 1);
    P(a) = P(a) + mpc.gen(i, 2);
end
P = P / 100;
Y=zeros(len_P, len_P);
for i=1:length(mpc.branch(:,1))
    a = mpc.branch(i, 1);
    b = mpc.branch(i, 2);
    if a > b
        Y(a, b) = 1 / sqrt(mpc.branch(i, 3) ^ 2+mpc.branch(i, 4) ^ 2);
    else
        Y(b, a) = 1 / sqrt(mpc.branch(i, 3) ^ 2+mpc.branch(i, 4) ^ 2);
    end
end
Y = Y + Y';
theta = mpc.bus(:, 9);
xlswrite('parameter\parameter14.xlsx', P, 'Sheet1')
xlswrite('parameter\parameter14.xlsx', Y, 'Sheet2')
xlswrite('parameter\parameter14.xlsx', theta, 'Sheet3')