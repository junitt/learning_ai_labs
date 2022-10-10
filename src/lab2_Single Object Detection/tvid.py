import os
from torch.utils import data
import transforms


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    def __init__(self, root, mode):
        self.samples=[]
        self.classes=['bird','car','dog','lizard','turtle']
        lst=[transforms.LoadImage(),transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.4,0.4,0.4))]
        self.transforms=transforms.Compose(lst)
        for i in range(len(self.classes)):
            imgs=os.path.join(root,self.classes[i])
            tags=os.path.join(root,self.classes[i]+'_gt.txt')
            with open(tags) as f:
                for line in f.readlines():
                    id,*parm=line.strip().split(' ')
                    if mode=='train'and int(id)>150:
                        break
                    elif mode=='test'and int(id)<=150:
                        continue
                    if int(id)>180:
                        break
                    self.samples.append({'path':os.path.join(imgs,'%06d.JPEG'%int(id)),'cls':i,'bbox':[int(num) for num in parm]})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        samp=self.samples[idx]
        image, label=self.transforms(samp['path'],samp['bbox'])
        return image,{'cls':samp['cls'],'box':label }
    ...

    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root='./data/tiny_vid', mode='train')
    import pdb; pdb.set_trace()
