%% Run this script to save cluster and PCA matrices.

% Set number of clusters
n_clusters = 9;

%% Read all trajectories
top_dir = '/media/annaochjacob/crucial/dataset/flexible/train/';
fruits = dir(top_dir);
fruits(ismember( {fruits.name}, {'.', '..'})) = [];  %remove . and ..
trajectories = [];

% Loop over fruits
disp('Reading trajectories...')
for f = 1:length(fruits)
    fruit = {fruits(f).name};
    fruit = fruit{1};
    trajectories_path = strcat(top_dir,fruit,'/trajectories.csv');
    t = csvread(trajectories_path);
    trajectories = [trajectories; t];
end

%% Compute variable mean for PCA reconstruction
means = mean(trajectories)
csvwrite('variable_mean.csv', means);

%% Perform k-means clustering
disp('Running k-means clustering algorithm...')
[idx,C] = kmeans(trajectories, n_clusters, 'MaxIter', 200);
csvwrite('centroids.csv', C) % Save cluster centroids
csvwrite('cluster_idx.csv', idx);

%% Perform PCA
disp('Finding principal components...')
coeff = pca(trajectories);
csvwrite('coeff.csv',coeff);

%% Plot clusters
disp('Plotting clusters...')
for c = 1:n_clusters
    c
    %if mod(c,20) == 1
    %    figure
    %end
    %subplot(4,5, mod(c,20)+1)
    subplot(3,3, c)
    hold on
   for i = 1:length(trajectories)
        if idx(i) == c && mod(i,10) == 0
            plot(trajectories(i,1:30),trajectories(i,31:60), 'b');
        end
    end
    p = plot(C(c,1:30),C(c,31:60),'r');
    axis([-25 25 0 30])
    p(1).LineWidth = 2;
end

saveas(gcf,'9_clusters.png')
saveas(gcf,'9_clusters.svg')
