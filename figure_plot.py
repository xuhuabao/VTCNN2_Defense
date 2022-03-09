import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches


# ###################### figure 8 (a) SNR = 10 (b) SNR=-10 #####################################
def figure8():
    eps_arr = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]
    attacksnr = -10
    figpath = 'output/Attack_Acc_{}.pdf'.format(attacksnr)

    data_normal = np.load('./output/Attack_Acc_normal_{}.npy'.format(attacksnr), allow_pickle=True)
    data_defense = np.load('./output/Attack_Acc_defense_{}.npy'.format(attacksnr), allow_pickle=True)
    data_normal = data_normal.item()
    data_defense = data_defense.item()

    clean_accs1 = data_normal['clean']
    fgm_accs1 = data_normal['fgm']
    mim_accs1 = data_normal['mim']
    pgd_accs1 = data_normal['pgd']

    clean_accs2 = data_defense['clean']
    fgm_accs2 = data_defense['fgm']
    mim_accs2 = data_defense['mim']
    pgd_accs2 = data_defense['pgd']

    xlen = [i for i in range(len(eps_arr))]
    plt.figure()

    p11, = plt.plot(xlen, fgm_accs1, marker='o', color='#FF5722')
    p12, = plt.plot(xlen, mim_accs1, marker='*', color='#9E9E9E')
    p13, = plt.plot(xlen, pgd_accs1, marker='+', color='#2196F3')
    p14, = plt.plot(xlen, clean_accs1, marker='^', color='black')
    l1 = plt.legend([p11, p12, p13, p14],
                    ['Normal VT-CNN2, FGSM',
                     'Normal VT-CNN2, MIM',
                     'Normal VT-CNN2, PGD',
                     'Normal VT-CNN2, No attack'],
                    loc='upper left')  # 10 , -10  loc='upper left'

    p21, = plt.plot(xlen, fgm_accs2, marker='o', color='#FF5722', linestyle='--')  # label='Defense VT-CNN2, FGSM'
    p22, = plt.plot(xlen, mim_accs2, marker='*', color='#9E9E9E', linestyle='--')  # label='Defense VT-CNN2, MIM',
    p23, = plt.plot(xlen, pgd_accs2, marker='+', color='#2196F3', linestyle='--')  # label='Defense VT-CNN2, PGD',
    p24, = plt.plot(xlen, clean_accs2, marker='^', color='black', linestyle='--')  # label='Defense VT-CNN2, No attack',

    l2 = plt.legend([p21, p22, p23, p24],
                    ['Defense VT-CNN2, FGSM',
                     'Defense VT-CNN2, MIM',
                     'Defense VT-CNN2, PGD',
                     'Defense VT-CNN2, No attack'],
                    loc='upper right' )  # 10 loc=(0.527, 0.47) , -10 loc='upper right'

    plt.gca().add_artist(l1)

    if attacksnr == 10:
        plt.ylim(0, 80)
    else:
        plt.ylim(0, 80)
    # plt.title(figpath)
    plt.xticks(xlen, eps_arr)
    plt.tick_params(labelsize=12)
    plt.xlabel('eps', {'size':12})
    plt.ylabel('Classification Accuracy(%)', {'size':12})
    plt.grid(axis='y', linestyle='-.', alpha=0.59)

    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()

def under_diff_snr_line():

    eps = 0.001
    figpth = 'output/Attack_Under_Diff_SNRs_line_{}.pdf'.format(eps)

    npypth1 = 'output/Attack_Under_Diff_SNRs_normal_line_{}.npy'.format(eps)
    npypth2 = 'output/Attack_Under_Diff_SNRs_defense_line_{}.npy'.format(eps)

    data1 = np.load(npypth1, allow_pickle=True).item()
    data2 = np.load(npypth2, allow_pickle=True).item()

    snrs = data1['snrs']
    xlen = [i for i in range(len(snrs))]

    clean_accs1 = data1['clean']
    fgm_accs1 = data1['fgm']
    mim_accs1 = data1['mim']
    pgd_accs1 = data1['pgd']

    clean_accs2 = data2['clean']
    fgm_accs2 = data2['fgm']
    mim_accs2 = data2['mim']
    pgd_accs2 = data2['pgd']

    p1, = plt.plot(xlen, fgm_accs1, marker='o', color='#FF5722')
    p2, = plt.plot(xlen, mim_accs1, marker='*', color='#9E9E9E')
    p3, = plt.plot(xlen, pgd_accs1, marker='+', color='#2196F3')
    p4, = plt.plot(xlen, clean_accs1, marker='^', color='#FFC107')
    l1 = plt.legend([p1, p2, p3, p4], ['Normal VT-CNN2, FGSM', 'Normal VT-CNN2, MIM', 'Normal VT-CNN2, PGD',
                                       'Normal VT-CNN2, No attack'], loc='upper left') # prop={'size':12}

    p5, = plt.plot(xlen, fgm_accs2, marker='o', color='#FF5722', label='FGSM', linestyle='--')
    p6, = plt.plot(xlen, mim_accs2, marker='*', color='#9E9E9E', label='MIM', linestyle='--')
    p7, = plt.plot(xlen, pgd_accs2, marker='+', color='#2196F3', label='PGD', linestyle='--')
    p8, = plt.plot(xlen, clean_accs2, marker='^', color='#FFC107', label='No attack', linestyle='--')
    l2 = plt.legend([p5, p6, p7, p8], ['Defense VT-CNN2, FGSM', 'Defense VT-CNN2, MIM', 'Defense VT-CNN2, PGD',
                                       'Defense VT-CNN2, No attack'], loc='upper right' ) # prop={'size':12}

    plt.gca().add_artist(l1)
    plt.xlabel('SNR', {'size':12})
    plt.ylabel('Classification Accuracy(%)', {'size':12})

    plt.xticks(xlen, snrs)
    plt.tick_params(labelsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='-.', alpha=0.59)
    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()

    # ###################
    figpth = 'output/Attack_Under_Diff_SNRs_bar_{}.pdf'.format(eps)
    under_diff_snr_bar(fgm_accs1, pgd_accs1, mim_accs1,
                       fgm_accs2, pgd_accs2, mim_accs2,
                       eps, figpth)


def under_diff_snr_bar(fgm_acc1, pgd_acc1, mim_acc1, fgm_acc2, pgd_acc2, mim_acc2, eps, figpth):

    num_list1 = [np.mean(fgm_acc1), np.mean(mim_acc1), np.mean(pgd_acc1)]  # Normal VTCNN2
    num_list2 = [np.mean(fgm_acc2), np.mean(mim_acc2), np.mean(pgd_acc2)]  # Defense VTCNN2

    width = 0.4
    x_tick = ['FGSM', 'MIM', 'PGD']
    title = "Average Accuracy under All SNRs with  ε = {}"
    normal_color = '#00939A'
    defense_color = '#F48333'

    # plt.title(title.format(eps))
    plt.xlabel("Attack Algorithm", {'size':12})
    plt.ylabel("Classification Accuracy(%)", {'size':12})
    if eps == 0.001:
        plt.ylim(15, 64)
    elif eps == 0.0015:
        plt.ylim(15, 64)

    bar1 = plt.bar(np.arange(len(num_list1))-width/2, num_list1, width=width, color=normal_color)
    plt.bar_label(bar1, fmt='%.4g', label_type='edge', fontproperties={'size':12})
    bar2 = plt.bar(np.arange(len(num_list2)) + width/2, num_list2, width=width, color=defense_color)
    plt.bar_label(bar2, fmt='%.4g', label_type='edge', fontproperties={'size':12})

    normal_patch = mpatches.Patch(label='Normal VT-CNN2', color=normal_color)
    defense_patch = mpatches.Patch(label='Defense VT-CNN2', color=defense_color)
    plt.legend(handles=[normal_patch, defense_patch])

    plt.gca().set_xticks(np.arange(len(num_list1)))  # x kedu weizhi
    plt.gca().set_xticklabels(x_tick)  # biaoqian
    plt.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()


if __name__ == '__main__':
    figure8()
    under_diff_snr_line()

