function [w theta] = ann(X,y,w,l,theta,beta,size2)
%X:row vector: sample
%y: sample label
%l:learning rate
%size2: size of nn, form:
  %[#of input, #of hidden neurons, #of output(1)]
syms f(a) a
f(a) = 1/(1+exp(-beta*a));

num_input = size2(1);
num_hidden = size2(2);
out_hidden = zeros(1,num_hidden);
err_hidden = zeros(1,num_hidden);
dt = zeros(size(theta));
%get hidden layer value
for i = 1:num_hidden
    id = num_input + i; % get neuron id
    otmp = vpa(X*w(1:num_input,id) + theta(id));
    out_hidden(i) = vpa(f(otmp));
end
out_hidden;

%output
id2 = sum(size2);
id = num_input + 1;
out_final = vpa(out_hidden*w(id:id2-1,id2) + theta(id2));
out_final = vpa(f(out_final));

%error
err_out = out_final*(1-out_final)*(y-out_final);
theta(id2) = theta(id2)+l*err_out;
for i = 1:num_hidden
    id = num_input + i; % get neuron id
    err_hidden(i) = err_out*out_hidden(i)*(1-out_hidden(i))*w(id,id2);
    %update w from this hidden neuron to output
    w(id,id2) = w(id,id2) + l*err_out*out_hidden(i);
    %update theta
    theta(id) = theta(id) + l*err_hidden(i);
    %update w from input to this hidden neuron
    for j = 1:num_input
        w(j,id) = w(j,id)+l*err_hidden(i)*X(j);
    end
end
err_hidden;
%print out
w;
theta;

end