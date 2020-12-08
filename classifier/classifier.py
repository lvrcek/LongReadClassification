import os, shutil, torch, json, argparse, csv
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

torch.manual_seed(99)
trainloss=[0,0]

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def normalize1(array):
    return array / torch.max(array)

def count(j):
    count=0
    index=[]
    for i, a in enumerate(real_labels):
        if a==j:
            count+=1
            index.append(i)
    return count, index

def dist(index_i):
    zero,one,two,three=0,0,0,0
    zero_list,one_list,two_list,three_list=[],[],[],[]
    for w in index_i:
        pred = top_class1[w]
        if pred==0:
            zero+=1
            zero_list.append(data['path'][w])
        elif pred==1:
            one+=1
            one_list.append(data['path'][w])
        elif pred==2:
            two+=1
            two_list.append(data['path'][w])
        elif pred==3:
            three+=1
            three_list.append(data['path'][w])
    return zero, one, two, three, zero_list, one_list, two_list, three_list

def evaluate():
    index_0, index_1, index_2, index_3 = count(0), count(1), count(2), count(3)
                    
    c_c, c_r, c_n, c_j = dist(index_0[1])[0:4]
    c_c_dir, c_r_dir, c_n_dir, c_j_dir = dist(index_0[1])[4:8] # actual_prediction
    r_c, r_r, r_n, r_j = dist(index_1[1])[0:4]
    r_c_dir, r_r_dir, r_n_dir, r_j_dir = dist(index_1[1])[4:8] # actual_prediction
    n_c, n_r, n_n, n_j = dist(index_2[1])[0:4]
    n_c_dir, n_r_dir, n_n_dir, n_j_dir = dist(index_2[1])[4:8] # actual_prediction
    j_c, j_r, j_n, j_j = dist(index_3[1])[0:4] 
    j_c_dir, j_r_dir, j_n_dir, j_j_dir = dist(index_3[1])[4:8] # actual_prediction
    
    chimeric_pred = [c_c, r_c, n_c, j_c]
    repetitive_pred = [c_r, r_r, n_r, j_r]
    regular_pred = [c_n, r_n, n_n, j_n]
    junk_pred = [c_j, r_j, n_j, j_j]

    return chimeric_pred, repetitive_pred, regular_pred, junk_pred

# P R E P R O C E S S I N G
def Preprocessing(array, length):
    '''
    This function increases the length of sequence by padding or duplicating
    each element (maximum seqeunce length//sequence length) times.
    Then, chooses random elements to duplicate to maximum length.
    '''
    padded=[]
    if len(array)<length:
        pad_num = length//len(array)
        padded = [array[i//pad_num] for i in range(len(array)*pad_num)]
        
        remainder = length-len(padded)
        idx = np.linspace(0,len(padded),num=remainder,dtype=int)
        for i in idx:
            padded.insert(i,padded[i])
    else:
        idx = np.linspace(0,len(array),num=length,dtype=int,endpoint=False)
        padded = [array[i] for i in idx]
    return torch.Tensor(padded)

class ArrayDataset(Dataset):
    def __init__(self, dataset_dir, length, transform=None):
        self.data = json.load(open(dataset_dir))
        self.path_list=[]
        self.array_list=[]
        self.label_list=[]
        self.length=length
        self.transform=transform
    
        for name in self.data:            
            self.array_list.append(np.array(self.data[name]['data_']))
            self.label_list.append(0)
            self.path_list.append(name)
                        
    def __len__(self):
        return len(self.array_list)
    
    def __getitem__(self, idx):
        if idx < len(self.path_list):
            array = Preprocessing(self.array_list[idx], self.length)
            array = array-np.median(array)
            label = torch.tensor(self.label_list[idx])
            path = self.path_list[idx]
            sample = {'array': array, 'label': label, 'path': path}
            return sample

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 10
        hidden_layers = [32*18, 32*16, 32*14, 32*12, 32*10, 32*8, 32*4, 32*2]
        kernel_size = [5, 5, 3, 3, 3, 3, 3, 2, 2]
        output_size = 1
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(input_size, hidden_layers[0], kernel_size[0], 1, 0, bias=False),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(hidden_layers[0], hidden_layers[1], kernel_size[1], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(hidden_layers[1], hidden_layers[2], kernel_size[2], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(hidden_layers[2], hidden_layers[3], kernel_size[3], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[3]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose1d(hidden_layers[3], hidden_layers[4], kernel_size[4], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[4]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # state size. (ngf*2) x 64
            nn.ConvTranspose1d(hidden_layers[4], hidden_layers[5], kernel_size[5], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[5]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # state size. (ngf*2) x 128
            nn.ConvTranspose1d(hidden_layers[5], hidden_layers[6], kernel_size[6], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[6]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # state size. (ngf*2) x 256
            nn.ConvTranspose1d(hidden_layers[6],hidden_layers[7], kernel_size[7], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[7]),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # state size. (ngf*2) x 512
            nn.ConvTranspose1d(hidden_layers[7], output_size, kernel_size[8], 2, 1, bias=False),
            nn.Dropout(p=0.5),
            nn.Tanh()
            # state size. (nc) x 1024
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1
        hidden_layers = [16, 16*2, 16*4, 16*8, 16*16, 16*32, 16*64]
        kernel_size = [3, 3, 5, 5, 7, 7, 7, 7]
        output_size = 16*128
        
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv1d(input_size, hidden_layers[0], kernel_size[0], 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf) x 16 x 16
            nn.Conv1d(hidden_layers[0], hidden_layers[1], kernel_size[1], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf*2) x 8 x 8
            nn.Conv1d(hidden_layers[1], hidden_layers[2], kernel_size[2], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf*4) x 4 x 4
            nn.Conv1d(hidden_layers[2], hidden_layers[3], kernel_size[3], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf*2) x 4 x 4
            nn.Conv1d(hidden_layers[3], hidden_layers[4], kernel_size[4], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            # 
            nn.Conv1d(hidden_layers[4], hidden_layers[5], kernel_size[5], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[5]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            #
            nn.Conv1d(hidden_layers[5], hidden_layers[6], kernel_size[6], 2, 1, bias=False),
            nn.BatchNorm1d(hidden_layers[6]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            #
            nn.Conv1d(hidden_layers[6], output_size, kernel_size[7], 2, 1, bias=False),
            nn.Dropout(p=0.2)
            # output size () x 2 x 2N
        )
        self.adv_layer = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(4096, 4 + 1))

    def forward(self, img):
        out = self.main(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class gan():        
    def __init__(self, data):
        self.data = data
        self.ds = ArrayDataset(self.data, 1405)
        self.batch_size=len(self.ds)
        
    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass
  
    def dataloader(self):
        self.dataloader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return self.dataloader

    def initialize(self, pretrained=True): 
        self.generator = Generator()
        self.generator = self.generator.to(device)
        self.discriminator = Discriminator()
        self.discriminator = self.discriminator.to(device)
        
        if pretrained==True:
            self.discriminator.load_state_dict(torch.load('discriminator_88.pth', map_location=lambda storage, loc: storage))
            self.generator.load_state_dict(torch.load('generator_88.pth', map_location=lambda storage, loc: storage))
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        
        self.dis_criterion = nn.CrossEntropyLoss()
        
        return self.generator, self.discriminator, self.optimizer_G, self.optimizer_D, self.dis_criterion

    def classify(self):
        with torch.no_grad():
            for i, d in enumerate(self.dataloader):
                real_images, real_path = d['array'], d['path']
                real_images = torch.unsqueeze(real_images, dim=1)
                z = torch.randn(real_images.shape[0], 10, 1)
    
                real_images, z = real_images.to(device), z.to(device)
                fake_images = self.generator(z)
                                
                valid_dis_r_validity, valid_dis_r_labels = self.discriminator(real_images)
                valid_dis_f_validity, valid_dis_f_labels = self.discriminator(fake_images)
                                            
                # E V A L U A T I O N ------------------------- S T A R T
                
                print("Classifying ...")
                
                top_ps1, top_class1 = valid_dis_r_labels.topk(1,dim=1)
                c_index=[i for i,_ in enumerate(top_class1) if _==0]
                self.c_name=[real_path[i] for i in c_index]
                r_index=[i for i,_ in enumerate(top_class1) if _==1]
                self.r_name=[real_path[i] for i in r_index]
                n_index=[i for i,_ in enumerate(top_class1) if _==2]
                self.n_name=[real_path[i] for i in n_index]
                j_index=[i for i,_ in enumerate(top_class1) if _==3]
                self.j_name=[real_path[i] for i in j_index]
                self.c=len(c_index)
                self.r=len(r_index)
                self.n=len(n_index)
                self.j=len(j_index)
                
        return self.c, self.r, self.n, self.j, self.c_name, self.r_name,self.n_name, self.j_name

    def txt(self):
        pred_chimeric=self.c_name
        f=open("chimerics.txt","w+")
        for node in pred_chimeric:
            row='ERR2173373.'+str(node)+'\n '
            f.write(row)
        f.close()

    def run(self):
        gan.dataloader()
        gan.initialize(pretrained=True)
        c, r, n, j, c_name, r_name, n_name, j_name = gan.classify()
        # gan.csv()
        gan.txt()

        print('Chimeric: {:}\nRepetitive: {:}\nRegular: {:}\nJunk: {:}\n'.format(c, r, n, j))

# ----------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""This script takes json files as input""")
    parser.add_argument('data', type=str, help='''Input directory of data.''')
    # parser.add_argument('--create_directories', action='store_true', help='''Creates directories for respective classified pileograms.''')
    args = parser.parse_args()
    gan = gan(args.data)
    
    with gan:
        gan.run()

    # if args.create_directories:
    #     old_dir=args.data[:-15]
    #     new_chimeric_dir=os.path.join(args.data,'gan_chimeric')
    #     os.system('mkdir {}'.format("'"+new_chimeric_dir+"'"))
    #     with open('chimerics.txt') as f:
    #         data=list(f.read().split('\n'))
    #         for row in data:
    #             node=str(row.split('.')[1])
    #             shutil.copy(os.path.join(old_dir,node+'.png'),os.path.join(new_chimeric_dir,node+'.png'))
