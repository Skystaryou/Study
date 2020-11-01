maxT=2;
maxX=1;
maxY=1;
deltaT=0.005;




deltaX=0.008;


Nt=maxT/deltaT+1;
Nx=maxX/deltaX+1;
u=zeros(Nx,Nx,Nt);




%initial condition for t=0
for i=1:Nx+1
    for j=1:Nx+1
        u(i,j,1)=0;
    end
end


%initial condition for t=maxT/Nt
for i=1:Nx+1
    for j=1:Nx+1
        u(i,j,2)=u(i,j,1)+deltaT*f((i-1)*deltaX)*f((j-1)*deltaX);
    end
end


%boundary condition
for t=1:Nt+1
    for i=1:Nx+1
        u(1,i,t)=0;
        u(i,1,t)=0;
        u(Nx+1,i,t)=0;
        u(i,Nx+1,t)=0;
    end
end


%implement the scheme
for t=3:Nt+1
    for i=2:Nx
        for j=2:Nx
            u(i,j,t)=2*u(i,j,t-1)-u(i,j,t-2)+((deltaT*deltaT)/(deltaX*deltaX))*(u(i+1,j,t-1)+u(i-1,j,t-1)+u(i,j+1,t-1)+u(i,j-1,t-1)-4*u(i,j,t-1));
        end
    end
end


%get the needed result
finalU=u(:,:,Nt+1);
figure;
surf(finalU)