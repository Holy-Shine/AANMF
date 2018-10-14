from AANMF import AANMF
from dataset import MovieRankDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from optparse import OptionParser
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# --------------- hyper-parameters------------------
user_max_dict_100k={
    'uid':944,  # 6040 users
    'gender':2,
    'age':9,
    'job':21
}

movie_max_dict_100k={
    'mid':1683,  # 3952 movies
}

user_max_dict_1m={
    'uid':6041,  # 6040 users
    'gender':2,
    'age':7,
    'job':21
}

movie_max_dict_1m={
    'mid':3953,  # 3952 movies
}





def evaluate(model,epoch,dataset='ml1m'):
    loss_function = nn.MSELoss()
    if dataset=='ml100k':
        datasets = MovieRankDataset(pkl_file='ml100k_test.p')
    else:
        datasets = MovieRankDataset(pkl_file='ml1m_test.pkl')
    dataloader = DataLoader(datasets,batch_size=1000,shuffle=False)  # one batch
    for batch in dataloader:
        with torch.no_grad():
            user_inputs = batch['user_inputs']
            movie_inputs = batch['movie_inputs']
            target = batch['target'].to(device)
            predicitons = model(user_inputs, movie_inputs)  # batch x score
            loss = loss_function(predicitons, target)
    str = "Epoch {} MSE: {}\n".format(epoch,loss)
    return str
def train_loss(model, epoch, dataset = 'ml1m'):
    loss_function = nn.MSELoss()
    if dataset=='ml100k':
        datasets = MovieRankDataset(pkl_file='ml100k_train.p')
    else:
        datasets = MovieRankDataset(pkl_file='ml1m_train.pkl')    
    losses = 0
    dataloader = DataLoader(datasets,batch_size=100,shuffle=False)  # one batch
    num = 0
    for batch in dataloader:
        with torch.no_grad():
            user_inputs = batch['user_inputs']
            
            batch_len = user_inputs['uid'].shape[0]
            num+=batch_len
            movie_inputs = batch['movie_inputs']
            target = batch['target'].to(device)
            predicitons = model(user_inputs, movie_inputs)  # batch x score
            loss = loss_function(predicitons, target)
            losses+=loss*batch_len
    epoch_loss = losses/num
    print("Epoch {} loss:{}".format(epoch,epoch_loss))
    str = "Epoch {} MSE: {}\n".format(epoch,epoch_loss)
    return str   
    
def train_eval(model,num_epochs=5, lr=0.001,batch_size=64,dataset='ml1m',loss_file='loss_file.txt',evaluate_file='evaluate_file.txt'):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    if dataset=='ml100k':
        datasets = MovieRankDataset(pkl_file='ml100k_train.p')
    else:
        datasets = MovieRankDataset(pkl_file='ml1m_train.pkl')
    dataloader = DataLoader(datasets,batch_size=batch_size,shuffle=True)

    eval_str=""
    loss_str=""
    for epoch in range(num_epochs):
        for i_batch,sample_batch in enumerate(dataloader):

            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = sample_batch['target'].to(device)

            model.zero_grad()

            tag_rank  = model(user_inputs, movie_inputs)

            loss = loss_function(tag_rank, target)
            if i_batch % 19 == 0:
                print("Epoch {}:{}".format(epoch,loss))
            loss.backward()
            optimizer.step()

        eval_str+=evaluate(model,epoch,dataset=dataset)
        loss_str+=train_loss(model,epoch,dataset=dataset)
    with open(evaluate_file,'w') as f:
        f.write(eval_str)
    with open(loss_file,'w') as f:
        f.write(loss_str)




def get_user_params():

    try:
        opt = OptionParser()
        opt.add_option('--lr',
                       dest='lr',
                       type=float,
                       help='set learning rate. default: 0.001',
                       default=0.001)
        opt.add_option('--batch_size',
                       dest='batch_size',
                       type=int,
                       help='set mini-batch\'s size. default: 64',
                       default=128)
        opt.add_option('--epoch',
                       dest='epochs',
                       type=int,
                       help='set training epoch. default: 20',
                       default=50)
        opt.add_option('--embed',
                       dest='embed_dim',
                       type=int,
                       help='set word vector dim. default: 8',
                       default=8)

        opt.add_option('--num_fc_layer',
                       dest='num_fc_layer',
                       type=int,
                       help='set number of fc layer, if None, simply dot 2 features. default: None',
                       default=None)
        opt.add_option('--attention',
                       dest='attention',
                       help='turn on attention mechanism. default: True',
                       default=True)
        opt.add_option('--dataset',
                       dest='dataset',
                       type=str,
                       help='select dataset. ml1m or ml100k. default: ml100k',
                       default='ml100k')
        opt.add_option('--loss_file',
                       dest='loss_file',
                       type=str,
                       help='name the loss output file',
                       default='loss_file')
        opt.add_option('--evaluate_file',
                       dest='evaluate_file',
                       type=str,
                       help='name the evaluate output file',
                       default='evaluate_file')
        opt.add_option('--model',
                       dest='model',
                       type=str,
                       help='SVD/NFM/AANMF/CUSTOM',
                       default='CUSTOM')
        (options, args) = opt.parse_args()

        
        embed_dim = options.embed_dim
        batch_size = options.batch_size
        epochs = options.epochs
        lr = options.lr
        num_fc_layer=  options.num_fc_layer
        attention = options.attention
        dataset = options.dataset
        loss_file = options.loss_file
        evaluate_file = options.evaluate_file
        if options.model=='SVD':
            num_fc_layer=None
            attention = False
        elif options.model=='NFM':
            num_fc_layer=1
            attention=False
        elif options.model=='AANMF':
            num_fc_layer=1
            attention = True

        user_parmas={'embed_dim':embed_dim,
                     'batch_size':batch_size,
                     'epochs':epochs,
                     'lr':lr,
                     'num_fc_layer':num_fc_layer,
                     'attention':attention,
                     'dataset':dataset,
                     'loss_file':loss_file,
                     'evaluate_file':evaluate_file}
        return user_parmas
    except:
        print('error')

def main():
    user_parms = get_user_params()
    embed_dim = user_parms['embed_dim']
    batch_size = user_parms['batch_size']
    epochs = user_parms['epochs']
    lr = user_parms['lr']
    num_fc_layer = user_parms['num_fc_layer']
    attention = user_parms['attention']
    dataset = user_parms['dataset']
    loss_file = user_parms['loss_file']
    evaluate_file = user_parms['evaluate_file']
    print('***********************training setting***************************')
    print('*dataset: {}'.format(dataset))
    print('*epoch: %d'%epochs)
    print('*batch_size: %d' % batch_size)
    print('*embed_dim: %d' % embed_dim)
    print('*learning_rate: %f' % lr)
    print('*number of fc layer: {}'.format(num_fc_layer))
    print('*attention: {}'.format(attention))
    print('*loss_file: {}'.format(loss_file))
    print('*evaluate_file: {}'.format(evaluate_file))
    print('******************************************************************')

    if dataset == 'ml100k':
        model = AANMF(user_max_dict=user_max_dict_100k, movie_max_dict=movie_max_dict_100k, num_fc_layer=num_fc_layer, attention=attention)
    else:
        model = AANMF(user_max_dict=user_max_dict_1m, movie_max_dict=movie_max_dict_1m, num_fc_layer=num_fc_layer,
                      attention=attention)
    model=model.to(device)
    print(model)
    # train and evaluate model

    train_eval(model=model, num_epochs=epochs,lr=lr,batch_size=batch_size,dataset=dataset, loss_file=loss_file,evaluate_file=evaluate_file)
    torch.save(model.state_dict(), 'Params/model_params.pkl')


if __name__=='__main__':

    main()


