from __future__ import print_function
from __future__ import absolute_import
from Omniglot import OmniglotNShotDataset
import gnog_test as gt
from options import Options
import torch
import pdb

import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
x_dtype = torch.float32
y_dtype = torch.int64

def new_batch(data, mode, rotate_flag):
    x_s, y_s, x_t, y_t = data.get_batch(str_type=mode, rotate_flag=rotate_flag)

    def squeze(x):
        for a in x:
            a.squeeze_(1)
    
    train_task = torch.tensor(x_s.swapaxes(2,4)).to(device, x_dtype)
    train_label = torch.tensor(y_s).to(device, y_dtype)
    test_task = torch.tensor(x_t.swapaxes(2,4)).to(device, x_dtype)
    test_label = torch.tensor(y_t).to(device, y_dtype)

    train_task = list(train_task.split(1,1))
    train_label = list(train_label.split(1,1))
    test_task = list(test_task.split(1,1))
    test_label = list(test_label.split(1,1))
    
    squeze(train_task)
    squeze(train_label)
    squeze(test_task)
    squeze(test_label)
    
    total_task = zip(train_task, train_label)
    task = zip(test_task, test_label)
    total_task.extend(task)
        
    return total_task


args = Options().parse()

data = OmniglotNShotDataset(args.data_root, batch_size=args.batch_size, 
        classes_per_set=args.N_way, samples_per_class=args.N_shot)

estimator = gt.Estimator(args.lr, args.img_channel, args.N_way, args.N_shot, args.N_test,
                args.num_updater_layer, args.num_kernel, args.hidden_size, args.edge_middle_size,
                args.node_middle_size, args.mean, args.std, args.save_root, args.use_cuda)

if __name__ == '__main__':
    for epoch in range(args.max_epoch):
        for batch in range(args.max_batch):
            total_task = new_batch(data, 'train', True)
            train_task = total_task[:(args.N_way * args.N_shot)]
            test_task = total_task[-args.N_test:]
            estimator.train_(train_task, test_task)
            
            if batch % args.eval_interval == 0:
                acc, loss = estimator.evaluate(test_task)
                estimator.display('meta-train', epoch, batch, acc, loss)
                # estimator.save_model_params(acc)

            if epoch % args.test_interval == 0 and batch % args.eval_interval == 0:
                total_task = new_batch(data, 'val', False)
                train_task = total_task[:(args.N_way * args.N_shot)]
                test_task = total_task[-args.N_test:]
                estimator.test_(train_task)
                acc, loss = estimator.evaluate(test_task)
                estimator.display('meta-validation', epoch, batch, acc, loss)
                
                total_task = new_batch(data, 'test', False)
                train_task = total_task[:(args.N_way * args.N_shot)]
                test_task = total_task[-args.N_test:]
                estimator.test_(train_task)
                acc, loss = estimator.evaluate(test_task)
                estimator.display('meta-test', epoch, batch, acc, loss)
                estimator.save_model_params(acc)


                


