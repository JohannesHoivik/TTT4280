import numpy as np
import pandas as pd
#------------------------------------------------------------------------------------------------------------------------------
#Basismålinger SNR reflektans
SNR_HF_basis_rod_refl = [11.074912990009382,
10.145391350218198,
9.621445813113215,
12.846830950221213,
10.805248473149655,
10.729011521952543,
11.434562109381542,
8.986697464784763]

SNR_HF_basis_gronn_refl = [7.396411468094976,
6.780099929035987,
7.875082264562061,
7.252622115555036,
7.741280445074679,
6.801803518443226]

SNR_HF_basis_blaa_refl = [7.2403716577104085,
4.353443351116495,
4.1165770865462585,
5.05977985054238,
5.726999896089369,
4.07832662540573,
6.059389079116788,
5.158832489844457]

basis_refl = [SNR_HF_basis_rod_refl, SNR_HF_basis_gronn_refl, SNR_HF_basis_blaa_refl]
#------------------------------------------------------------------------------------------------------------------------------
#Basismålinger SNR transmittans
SNR_HF_basis_rod_tran= [6.795833381160507,
7.148658150253114,
4.8028970052991395,
7.740347448099732,
7.9884613947801695,
4.744448567047058,
8.05868950594908,
7.459671758272393]

SNR_HF_basis_gronn_tran = [9.558508393132843,
8.689773857941034,
5.892312302083193,
8.901476686929257,
9.167638816990369,
6.486501608353032,
9.072268598631322,
8.880062088141283]

SNR_HF_basis_blaa_tran = [7.452798616892791,
4.128883652558151]

basis_tran = [SNR_HF_basis_rod_tran, SNR_HF_basis_gronn_tran, SNR_HF_basis_blaa_tran]
#------------------------------------------------------------------------------------------------------------------------------
#Robusthetstest SNR kaldfinger transmittans
SNR_HF_kaldfinger_rod = [4.804261568340756,
7.301678534943631,
7.1138167118125,
6.0821375595909055,
4.3385153219527]

SNR_HF_kaldfinger_gronn = [7.689375301987682,
10.458311616549098,
7.634923945657023,
6.775613056516333,
8.620385669218505,
9.549863548711986,
7.688122010035418,
4.975298778352229]

SNR_HF_kaldfinger_blaa = [5.234059780506895,
5.773505289471384]

kaldfinger = [SNR_HF_kaldfinger_rod, SNR_HF_kaldfinger_gronn, SNR_HF_kaldfinger_blaa]
#------------------------------------------------------------------------------------------------------------------------------
#Robusthetstest SNR varmfinger transmittans
SNR_HF_varmfinger_rod = [7.540181533239002,
5.3769489633767185,
6.405206794268381,
7.033899815787768,
5.562037709297973,
6.681257729929087,
6.760055143592043,
5.042329143729064]

SNR_HF_varmfinger_gronn = [8.509829133186638,
7.483718355318391,
7.681885967887388,
7.453653464862409,
5.088920430874032,
8.428854305497543,
7.284595771974866,
6.967738890605211]

SNR_HF_varmfinger_blaa = [7.611898785502992,
5.671710643811624,
5.2648610848331305,
6.1111772261679125,
7.445216331788448]

varmfinger = [SNR_HF_varmfinger_rod, SNR_HF_varmfinger_gronn, SNR_HF_varmfinger_blaa]
#------------------------------------------------------------------------------------------------------------------------------
#skriving av SNR-beregninger til csv fil
sample_size_list = []
mean_list = []
std_list = []

for element in basis_refl:
    sample_size_list.append(len(element))
    mean_list.append(np.mean(element))
    std_list.append(np.std(element))
for element in basis_tran:
    sample_size_list.append(len(element))
    mean_list.append(np.mean(element))
    std_list.append(np.std(element))
for element in kaldfinger:
    sample_size_list.append(len(element))
    mean_list.append(np.mean(element))
    std_list.append(np.std(element))
for element in varmfinger:
    sample_size_list.append(len(element))
    mean_list.append(np.mean(element))
    std_list.append(np.std(element))


maalinger = {
    "Sample-size" : sample_size_list,
    "Mean SNR" : mean_list,
    "Std SNR" : std_list
}

df = pd.DataFrame(maalinger)
df.to_csv("Lab3/SNR_beregnigner_med_preprosessering.csv",index=False)

#------------------------------------------------------------------------------------------------------------------------------
#Skriving av SNR-beregninger til terminal
""" print("Basismålinger - reflektans:")
for element in basis_refl:
    print("-------------------------------------------------------------------------")
    print(f"    Målingene har en sample-size på: {len(element)}")
    print(f"    Mean-estimate: {np.mean(element)}")
    print(f"    Std-estimate: {np.std(element)}")
print("--------------------------------------------------------------------------------------------------------------------------")
print("Basismålinger - transmittans:")
for element in basis_tran:
    print("-------------------------------------------------------------------------")
    print(f"    Målingene har en sample-size på: {len(element)}")
    print(f"    Mean-estimate: {np.mean(element)}")
    print(f"    Std-estimate: {np.std(element)}")
print("--------------------------------------------------------------------------------------------------------------------------")
print("Robusthetstest - kald finger:")
for element in kaldfinger:
    print("-------------------------------------------------------------------------")
    print(f"    Målingene har en sample-size på: {len(element)}")
    print(f"    Mean-estimate: {np.mean(element)}")
    print(f"    Std-estimate: {np.std(element)}")
print("--------------------------------------------------------------------------------------------------------------------------")
print("Robusthetstest - varm finger:")
for element in varmfinger:
    print("-------------------------------------------------------------------------")
    print(f"    Målingene har en sample-size på: {len(element)}")
    print(f"    Mean-estimate: {np.mean(element)}")
    print(f"    Std-estimate: {np.std(element)}") """