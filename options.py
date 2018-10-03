import argparse

class Options():
    def __init__(self):

        parser = argparse.ArgumentParser(description= 'test_all')

        parser.add_argument('--lr', type = list, default = [0.0001, 0.001, 0.001],
                    help = 'learning rate list for ...')
        parser.add_argument('--max_epoch', type = int, default = 1000000, 
                    help = 'total training epoch')
        parser.add_argument('--max_batch', type = int, default = 100, 
                    help = 'total traing batch number per epoch')
        parser.add_argument('--data_root', type = str, default = './omniglot',
                    help = 'root directory to place Omniglot')
        parser.add_argument('--save_root', type = str, default = './save_model',
                    help = 'root directory to save the model params')
        parser.add_argument('--batch_size', type = int, default = 32, 
                    help = 'batch size for training and testing')
        parser.add_argument('--N_way', type = int, default = 5,
                    help = 'Number of classes in meta-train routine')
        parser.add_argument('--N_shot', type = int, default = 1,
                    help = 'Number of samples of each class in meta-train routine')
        parser.add_argument('--N_test', type = int, default = 1,
                    help = 'Number of classes(one sample each class) in meta-test routine')
        parser.add_argument('--num_updater_layer', type = list, 
                    default = [2, 2, 1, 1], 
                    help = 'elem1 for num of node updater layer. elem2 for node output, elem3 for task edge, elem4 for layer edge')
        parser.add_argument('--hidden_size', type = int, default = 256,
                    help = 'hidden size of lstm updater in every node')
        parser.add_argument('--edge_middle_size', type = list, default = [],
                    help = 'middle size of edge mlp updater. Note: the len of the list must equal to the value set in num_updater_layer')
        parser.add_argument('--node_middle_size', type = list, default = [512],
                    help = 'middle size of node output layer. Note: the len of the list must equal to the value set in num_updater_layer')
        parser.add_argument('--mean', type = float, default = 0.,
                    help = 'mean of Gaussian distribution to generate grads of the test sample')
        parser.add_argument('--std', type = float, default = 1.,
                    help = 'std of Gaussian distribution to generate grads of the test sample')
        parser.add_argument('--img_channel', type = int, default = 1, 
                    help = 'the channel size of the input image')
        parser.add_argument('--use_cuda', type = bool, default = True,
                    help = 'use cuda or not')
        parser.add_argument('--eval_interval', type = int, default = 50,
                    help = 'interval for evaluating the model, printing the res and saving the best. ')
        parser.add_argument('--test_interval', type = int, default = 50,
                    help = 'interval for validating and testing the model') 
        parser.add_argument('--num_kernel', type = int, default = 64,
                    help = 'the number of kernels in every conv layer')
        
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
