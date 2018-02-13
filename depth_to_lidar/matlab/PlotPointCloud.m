points = csvread('hej.csv');
scatter3(points(:,1),points(:,2),points(:,3),'.');
set(gca,'Color','k');