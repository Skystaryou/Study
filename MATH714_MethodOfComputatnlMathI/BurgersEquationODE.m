maxT=10;
minT=0;
minX=0;
maxX=1;
deltaX=0.05;
deltaT=0.001;

Nt=maxT/deltaT+1;
Nx=maxX/deltaX+1;

u=zeros(Nt,Nx);

% initial condition
for i=1:Nx
    u(1,i)=1.5+sin(2*pi*(i-1)*deltaX);
end

% boundary condition (PBC)
for i=1:Nt
    u(i,1)=1.5+sin(2*pi*(i-1)*deltaT);
    u(i,Nx)=1.5+sin(2*pi*(i-1)*deltaT);
end

%implement the scheme
for i=2:Nt
    for j=2:(Nx-1)
        u(i,j)=u(i-1,j)+(deltaT/(2*deltaX))*u(i-1,j)*(u(i-1,j+1)-u(i-1,j-1));
    end
end

%get the needed result
finalU=u(Nt,:);
%plot(finalU)
surf(u)