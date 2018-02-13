points = csvread('../python/trimmed_point_cloud.csv');
scatter3(points(:,1),points(:,2),points(:,3),'.');
set(gca,'Color','k');