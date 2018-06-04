n_clusters = 20;

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

disp('Running k-means clustering algorithm...')
[idx,C] = kmeans(trajectories, n_clusters, 'MaxIter', 100);
save C C % Save cluster centroids

disp('Finding principal components...')
coeff = pca(trajectories);
save coeff coeff

disp('Plotting clusters...')
for c = 1:n_clusters
    c
    if mod(c,20) == 1
        figure
    end
    subplot(4,5, mod(c,20)+1)
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
