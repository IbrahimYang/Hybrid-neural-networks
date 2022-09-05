import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
# import seaborn as sns
channel=256
samples=800
task_num=40
import torch
save_path=''
from tqdm import tqdm
def PCA_reduction(feature,dimension):
    pca = PCA(dimension)
    pca.fit(feature)
    print(pca.explained_variance_ratio_)
    print('PCA sum:', np.sum(pca.explained_variance_ratio_))
    # plt.stem(pca.explained_variance_ratio_)
    # plt.show()
    feature_new = pca.transform(feature)
    print('PCA shape:',feature_new.shape)
    return feature_new
def get_kmeans_score(X,clusters):
    pred=KMeans(clusters).fit_predict(X)
    score=metrics.calinski_harabaz_score(X,pred)
    return score
def scatter_show(feature_new):
    plt.scatter(feature_new[:1000,0],feature_new[:1000,1])
    plt.show()
def clustering_test(feature_new):
    result = []
    clusters = np.linspace(2, 200, 19, dtype=np.int32)
    for i in clusters:
        score = get_kmeans_score(feature_new, i)
        result.append(score)
        print(i, score)

    plt.plot(clusters, result)
    plt.grid()
    plt.show()
def tSNE_show_old(gate):
    np.random.seed(1)
    feature=gate
    label=[[i]*samples for i in range(task_num)]
    tsne=TSNE(n_components=2)
    tsne.fit(feature)
    feature_new=tsne.embedding_
    print('tSNE shape:',feature_new.shape)
    plt.scatter(x=feature_new[:,0],y=feature_new[:,1],c=label,marker='.',linewidths=0)
    plt.colorbar()
    plt.savefig(save_path + 'gate_tsne.png', dpi=200)
    plt.close()

def tSNE_show(gate):
    np.random.seed(1)
    feature=gate
    label=[[i]*samples for i in range(task_num)]
    label=sum(label,[])
    tsne=TSNE(n_components=2)
    tsne.fit(feature)
    feature_new=tsne.embedding_
    print('tSNE shape:',feature_new.shape)

    feature_new_1=[]
    feature_new_2=[]
    label_1=[]
    label_2=[]

    for i in range(len(label)):
        if(label[i]%4==1 or label[i]%4==2):
            if(np.random.rand()<0.02):
                feature_new_1.append(feature_new[i])
                label_1.append(label[i])
        else:
            if (np.random.rand()< 0.02):
                feature_new_2.append(feature_new[i])
                label_2.append(label[i])
    feature_new_1=np.array(feature_new_1)
    feature_new_2=np.array(feature_new_2)

    print('training size:{},testing size:{}'.format(len(label_1),len(label_2)))
    plt.scatter(x=feature_new_1[:,0],y=feature_new_1[:,1],c=label_1,marker='.',linewidths=0.1,label='train',s=70,cmap='jet')
    plt.scatter(x=feature_new_2[:, 0], y=feature_new_2[:, 1], c=label_2, marker='v', linewidths=0.1,label='test',s=70,cmap='jet')
    plt.axis([-70,70,-70,70])
    fs=11
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.colorbar()
    plt.legend()
    plt.savefig('2020_10_29/gate_tsne.png', dpi=1000)
    plt.close()

def get_cosine_similarity(gate):
    eps=1e-10
    inner_product=gate.mm(gate.t())
    f_norm=torch.norm(gate,p='fro',dim=1,keepdim=True)
    outter_product=f_norm.mm(f_norm.t())
    cosine=inner_product/(outter_product+eps)
    return cosine.cpu().numpy()

def calculate_similarity(data):
    result=get_cosine_similarity(torch.FloatTensor(data).cuda())[:32,:32]
    print(result.shape)
    plt.imshow(result, interpolation='nearest',vmin=0,vmax=1,cmap='jet')
    plt.colorbar()
    plt.savefig('sim.png')
    plt.show()
    return result

def show_gate(gate_array,title):
    plt.imshow(gate_array, interpolation='nearest',vmin=0,vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.show()

def main_1():
    root_path = 'xdg_128task_16384chan'
    path =root_path + '/gate_to_train.npy'
    print(np.load(path).shape)
    data = np.load(path).reshape(128, 16384*2)

    # data=np.mean(data, axis=1) > 0.5
    print(data.mean())
    print(data.shape)
    test_gate_s = calculate_similarity(data)
    print(test_gate_s)

    # plt.figure(facecolor="white")
    # plt.imshow(np.mean(data, axis=1)[:, 0:128], interpolation='nearest',vmin=0,vmax=1,cmap='jet')
    # plt.xticks(np.arange(0,128,32),labels=np.arange(0,128,32),fontsize=13)
    # plt.yticks(np.arange(0,40,10),labels=np.arange(0,40,10),fontsize=13)
    # cbar_ax = plt.gcf().add_axes([0.92, 0.333, 0.02, 0.324])
    # plt.colorbar(cax=cbar_ax)
    # plt.savefig('2020_10_29/gate_scatter.png', dpi=1000)
    # plt.close()
    #
    # np.save('2020_10_29/gate_scatter.npy',np.mean(data, axis=1))


    # gate_label = np.load(save_path+root_path + '/p_similarity.npy')[-task_num:, -task_num:]
    # print(gate_label.shape)
    # plt.figure(facecolor="white")
    # plt.imshow(gate_label, interpolation='nearest',vmin=0,vmax=1)
    # plt.colorbar()
    # plt.savefig(save_path + 'task_similarity.png', dpi=1000)
    # np.save(save_path + 'task_similarity.npy',gate_label)
    # plt.close()

    feature_new = PCA_reduction(data.reshape(-1, channel * 2), 1024)
    tSNE_show(feature_new)



if __name__ == '__main__':
    main_1()
