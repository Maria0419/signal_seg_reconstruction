clear; close all;

rank = 150;

filename = '../artery_segmentation/nfold1c2f16_out/loss.csv';
loss_rec = readtable(filename);

loss_sunet4c = readtable('../signal_segmentation/nfold1c4_out/loss.csv');
loss_sunet4c = sortrows(loss_sunet4c, 'Loss', 'descend');
id = loss_sunet4c.Order(rank);

idx = load('indices.mat');
nfold1 = sort(idx.parts{2});

path_og = sprintf('../signal_segmentation/dataset_interpooled/img_%04d_image.png',nfold1(id));
og = imread(path_og);

path_label = sprintf('../signal_segmentation/dataset_interpooled/img_%04d_label.png',nfold1(id));
label = imread(path_label);

path_outrec = sprintf('../artery_segmentation/nfold1c2f16_out/output%04d.png',id);
out_rec = imread(path_outrec);

path_rec = sprintf('../artery_segmentation/dataset_reconstructed/img_%04d_reconstructed.png',nfold1(id));
rec = imread(path_rec);

path_sunet4c = sprintf('../signal_segmentation/nfold1c4_out/output%04d.png',id);
out_sunet4c = imread(path_sunet4c);

path_ddimg = sprintf('nfold1_out/output%04d_image.png',id);
ddimg = imread(path_ddimg);

path_ddsig = sprintf('nfold1_out/output%04d_label.png',id);
ddsig = imread(path_ddsig);

threshold = 220; % Adjust if necessary
label_bin = label > threshold;
out_sunet4c = out_sunet4c > threshold;
out_rec = out_rec > threshold;
out_ddsig = ddsig > threshold; 

result_img = zeros(size(label, 1), size(label, 2), 3, 'uint8');

mask_red = out_rec & ~label_bin;
result_img(:, :, 1) = mask_red * 200; 
mask_blue = label_bin & ~out_rec;
result_img(:, :, 3) = mask_blue * 255; 
mask_green = label_bin & out_rec;
result_img(:, :, 2) = mask_green * 200; 

result_sig = zeros(size(label, 1), size(label, 2), 3, 'uint8');

mask_red = out_sunet4c & ~label_bin;
result_sig(:, :, 1) = mask_red * 200; 
mask_blue = label_bin & ~out_sunet4c;
result_sig(:, :, 3) = mask_blue * 255; 
mask_green = label_bin & out_sunet4c;
result_sig(:, :, 2) = mask_green * 200; 

result_ddsig = zeros(size(label, 1), size(label, 2), 3, 'uint8');

mask_red = out_ddsig & ~label_bin;
result_ddsig(:, :, 1) = mask_red * 200;
mask_blue = label_bin & ~out_ddsig;
result_ddsig(:, :, 3) = mask_blue * 255;
mask_green = label_bin & out_ddsig;
result_ddsig(:, :, 2) = mask_green * 200;


figure;
subplot(2,3,1); imshow(og); title(sprintf('Original Image %d',nfold1(id)));
subplot(2,3,2); imshow(rec); title(sprintf('Reconstructed %d',nfold1(id)));
subplot(2,3,3); imshow(ddimg); title(sprintf('UNET Reconstruction %d',nfold1(id)));
subplot(2,3,4); imshow(label); title('Label Image');
%subplot(2,3,5); imshow(result_img); title(sprintf('Img UNET-DiceL: %.04f',loss_rec.Loss(id)));
subplot(2,3,5); imshow(result_sig); title(sprintf('Sig UNET-DiceL: %.04f',loss_sunet4c.Loss(rank)));
subplot(2,3,6); imshow(result_ddsig); title(sprintf('DDSig UNET-DiceL:'));

disp(loss_sunet4c.Order(rank));