not_finished=true;
now_error=-1;
N=9;

while not_finished
    N=N+1;
    x_true=0:1/N:1;
    y_true=linspace(0,0,N+1);

    sim_N=100000;
    x_sim=0:1/sim_N:1;
    for i= 1: N+1
        y_true(i)=f(x_true(i));
    end
    y_sim=interp1(x_true,y_true,x_sim);


    uniform_norm_error=-1.0;
    for i=1:sim_N+1
        if (abs(y_sim(i)-f(x_sim(i)))>uniform_norm_error)
            uniform_norm_error=abs(y_sim(i)-f(x_sim(i)));
        end
    end

    if (uniform_norm_error<=0.01)
        not_finished=false;
    end
    
    now_error=uniform_norm_error;
    
end

now_error

N
