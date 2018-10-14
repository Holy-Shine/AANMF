import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AANMF(nn.Module):

    def __init__(self, user_max_dict, movie_max_dict, embed_dim=8, fc_size=8, num_fc_layer=None, attention=False):
        '''

        Args:
            user_max_dict: the max value of each user attribute. {'uid': xx, 'gender': xx, 'age':xx, 'job':xx}
            user_embeds: size of embedding_layers.
            movie_max_dict: {'mid':xx}
            fc_sizes: fully connect layer sizes. default=embedding_size
            num_fc_layer: the number of fc layer, if None, simply dot two feature
            attention: whether turn on attention
        '''

        super(AANMF, self).__init__()

        # save params to model
        self.embed_dim = embed_dim
        self.num_fc_layer = num_fc_layer
        self.attention = attention


        #------------------------------------------- embedding part ------------------------------------------
        # user embeddings
        self.embedding_uid = nn.Embedding(user_max_dict['uid'], embed_dim)
        self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim)
        self.embedding_age = nn.Embedding(user_max_dict['age'], embed_dim)
        self.embedding_job = nn.Embedding(user_max_dict['job'], embed_dim)

        # item embeddings
        self.embedding_mid = nn.Embedding(movie_max_dict['mid'], embed_dim)


        #------------------------------------------- attention part ------------------------------------------
        self.att_linear = nn.Linear(embed_dim*2, embed_dim)
        self.att_softmax = nn.Softmax(dim=1)


        #------------------------------------------- fc part ------------------------------------------------------
        if num_fc_layer!=None:
            self.fc_layer = nn.ModuleList([nn.Linear(embed_dim,embed_dim),nn.ReLU()]*num_fc_layer)
            if torch.cuda.is_available():
                self.fc_layer.to(device)

            self.out_layer = nn.Linear(embed_dim,1)


        # BatchNorm layer
        self.BN = nn.BatchNorm2d(1)



    def attention_cell(self, embed_attribute, embed_item):
        '''Process of Attention Cell in AANMF framework

        Args:
            embed_attribute: embed of one of user's attribute
            embed_item: embed of item
        '''
        V = torch.cat([embed_item, embed_attribute],dim=1)
        v = self.att_linear(V)
        lambdda = self.att_softmax(v)

        return lambdda


    def pairwise_pooling(self, att_set, embed_uid):
        '''Process of Pairwise Pooling

        Args:
            att_set: a list of attribute embed after attention
            embed_uid: embed of user
        '''
        I_u = torch.stack(att_set)
        I_u = I_u.transpose(0,1) # 3*batch_size*embed_dim --> batch_size*3*embed_dim
        tilde_a = I_u.sum(dim=1)
        minus_a = torch.sum(I_u*I_u,dim=1)

        p_u = 1./2*((embed_uid+tilde_a)*(embed_uid+tilde_a)-embed_uid*embed_uid-minus_a)

        return p_u


    def forward(self, user_input, movie_input):
        # pack train_data
        uid = user_input['uid']
        gender = user_input['gender']
        age = user_input['age']
        job = user_input['job']

        mid = movie_input['mid']


        if torch.cuda.is_available():
            uid, gender, age, job,mid = \
            uid.to(device).squeeze(), gender.to(device).squeeze(), age.to(device).squeeze(), job.to(device).squeeze(), mid.to(device).squeeze()


        # ---------------------------------Embedding Layer ----------------------------------------------------------------
        embedding_uid    = self.embedding_uid(uid)
        embedding_gender = self.embedding_gender(gender)
        embedding_age    = self.embedding_age(age)
        embedding_job    = self.embedding_job(job)

        embedding_mid    = self.embedding_mid(mid)

        # ---------------------------------Attention Layer------------------------------------------------------------------
        lambda_g = self.attention_cell(embedding_gender, embedding_mid) if self.attention else 1
        lambda_a = self.attention_cell(embedding_age, embedding_mid) if self.attention else 1
        lambda_j = self.attention_cell(embedding_job, embedding_mid) if self.attention else 1


        # ----------------------------------Pooling Layer----------------------------------------------------------------------
        feature_user = self.pairwise_pooling([lambda_g*embedding_gender, lambda_a*embedding_age, lambda_j*embedding_job], embedding_uid)
        feature_item = embedding_mid


        # ----------------------------------Fully Connect Layer-----------------------------------------------------------------
        if self.num_fc_layer!=None:
            out = feature_item*feature_user
            for l in self.fc_layer:
                out = l(out)

            out = self.out_layer(out)


        else:
            out = torch.bmm(feature_user.view(-1,1,self.embed_dim), feature_item.view(-1,self.embed_dim,1)).squeeze(1)

        return out








