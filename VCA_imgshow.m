clc; clear; close all;

%%
x_line = 175;
lambda = 51:1005;

frame = 1;

b = load('Simu_data21_5s_Gau0.4_bestModel.mat');

raw_data_1 = b.data; %220x175x955 double


d = load('wn_calibrated_1005_new.mat');
wn_cali = d.wn;
wavlen = wn_cali;


raw_data_2 = permute(raw_data_1, [3 1 2]); %955x220x175

figure(1)
plot(raw_data_2(:, 40, 65))

[s1 s2 s3] = size(raw_data_2);
for ii = 1:s2
    for jj = 1:s3
        raw_data_3(:,ii,jj) = raw_data_2(:,ii,jj)./sum(raw_data_2(:,ii,jj), 'all');
    end
end
raw_data_2 = raw_data_3;

raw_data_2 = double(raw_data_2);

% 
figure(2)
imshow(squeeze(raw_data_2(:, :, 5))', []);

%%
[z_lambda x_row y_col] = size(raw_data_2);
x_n = reshape(raw_data_2, [z_lambda x_row*y_col]);

%% VCA
[Ae, Index, Rp] = vca(x_n, 'Endmembers',4,... % 'SNR',1000,...
                                    'verbose','off');
                                

test_data = x_n';

% find the weights from the rest of the image
for ij = 1:size(test_data,1)
    a=test_data(ij,:)';
    b=lsqnonneg(Ae,test_data(ij,:)');
    weight(ij,:) =b;
end

%%
rbg_show = zeros(x_row, y_col, 3);
tmpI = reshape(weight, x_row,y_col, size(weight, 2));

if size(weight,2) <= 3
    rbg_show(:, :, 1:size(tmpI, 3)) = tmpI;
else
    rbg_show(:,:,1) = tmpI(:,:,1) + 0.5*tmpI(:,:,4);
    rbg_show(:,:,2) = tmpI(:,:,2);
    rbg_show(:,:,3) = tmpI(:,:,3) + 0.5*tmpI(:,:,4);
end

colorNames = {'red', 'green', 'blue', 'magenta', 'gold', 'black', 'yellow', 'cyan'};
colorVals = {'r', 'g', 'b', 'm', [255 215 0]./255, 'k', 'y', 'c'};

figure(4);clf;
hold on
subplot(2, size(weight, 2)+1, 1:(size(weight, 2)+1))
h = plot( Ae, 'linewidth', 2);
for j = 1:size(weight, 2)
    set(h(j), 'Color', colorVals{j});
end

ylabel('intensity (a.a.)')
title('VCA restored spectra')
set(gca, 'fontsize', 20)


subplot(2, size(weight, 2)+1, size(weight, 2)+2)
imshow(flip(rbg_show(:, :, :), 2), [])
title('Pseudo color')
for ij = 2:(1+size(weight, 2))
    subplot(2, size(weight, 2)+1, ij+size(weight, 2)+1)
    imshow(flip(tmpI(:, :, ij-1), 2), [])
    title(['Layer ', num2str(ij-1), ': ', colorNames{ij-1}])
end

%%
choice = [1 4 1];
rgb = zeros(size(tmpI, 1), size(tmpI, 2), size(choice,2));
rgb(:,:,1) = tmpI(:, :, choice(1));
% rgb(:,:,2) = zeros(size(tmpI, 1), size(tmpI, 2));
rgb(:,:,2) = tmpI(:, :, choice(2));
% rgb(:,:,3) = zeros(size(tmpI, 1), size(tmpI, 2));
rgb(:,:,3) = tmpI(:, :, choice(3));
% rgb(:, :, 1:(end-3+length(choice))) = flip(tmpI(:, :, choice), 2);

figure(5);clf
imshow(rgb, [], 'border', 'tight', 'initialmagnification', 'fit');
set(gcf, 'Position', [700, 450, 512, 512]);
set(gcf, 'Position', [700, 450, 512, 512]);
set(gcf,'position',[1200 400 175*3 210*3])
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca, 'looseInset', [0 0 0 0])


%%
figure(6);clf;
 plot( wn_cali(lambda),Ae(:, choice(1)), 'r','linewidth',3);
hold on
plot( wn_cali(lambda), Ae(:, choice(2)), 'g','linewidth',3);
hold on
plot(wn_cali(lambda), Ae(:, choice(3)), 'b','linewidth',3); % wn(rr), Ae(:, choice(1)), 'r', 
xlim([min(wn_cali(lambda)) max(wn_cali(lambda))])
ylabel('intensity (a.a.)')
xlabel('Raman shift (cm^{-1})')
title('VCA restored spectra')
set(gca, 'fontsize', 20)

