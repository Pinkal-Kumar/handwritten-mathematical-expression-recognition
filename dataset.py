import torch
import time
import pickle as pkl
import torchvision
from torch.utils.data import DataLoader, Dataset, RandomSampler
import numpy


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def pad_img(self,img):
        # create a sample image of size (1,h,w)
        image = img  # sample image of size (1,100,200)

        # set the desired final size of the image
        final_size = (1, 481, 2116)

        # calculate the amount of padding required for each dimension
        h_padding = max((final_size[1] - image.shape[1]), 0)
        w_padding = max((final_size[2] - image.shape[2]), 0)
        top = h_padding // 2
        bottom = h_padding - top
        left = w_padding // 2
        right = w_padding - left

        # pad the image with zeros to the desired final size
        padded_image = numpy.pad(image, ((0,0), (top,bottom), (left,right)), mode='constant', constant_values=0)

        # check the size of the padded image
        return padded_image   # output: (1, 481, 2116)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        #image = self.pad_img(image)
        image = torch.Tensor(255-image) / 255
        image = torchvision.transforms.Resize(size = (64,256))(image)
        print("size: ", image.shape)
        image = image.unsqueeze(0)
        #labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"training data path images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"Verify data path images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[1]
    proper_items = []
    for item in batch_images:
        ch, ht, wd = item[0].shape[1], item[0].shape[2], item[0].shape[3]
        item_res = torch.reshape(item[0], (ch, ht, wd))
        if item_res.shape[1] * max_width > 1600 * 320 or item_res.shape[2] * max_height > 1600 * 320:
            continue
        max_height = item_res.shape[1] if item_res.shape[1] > max_height else max_height
        max_width = item_res.shape[2] if item_res.shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append((item_res, item[1]))

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'common {len(words)} class symbol.')
        self.words_dict = {words[i].strip().split()[0]: i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)].split()[0] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}
