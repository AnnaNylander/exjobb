data = csvread('/Users/Jacob/Desktop/result.csv',1,0);
x = data(:,1);
y = data(:,2);
z = data(:,3);
intensity = data(:,4);
color = jet(256);
size = 4;
roi = 60;
scatter(x,y,size,color(intensity+1,:),'filled');
set(gca,'Color','k'); % Set background color
pbaspect([1 1 1]); % Set aspect ratio of plot box
ylim([-roi roi]);
xlim([-roi roi]);