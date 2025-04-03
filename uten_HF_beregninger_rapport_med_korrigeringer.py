import numpy as np
import pandas as pd
#------------------------------------------------------------------------------------------------------------------------------
#Basismålinger SNR reflektans
SNR_uten_HF_basis_rod_refl = [11.865779379965138,
9.82,
8.99,
10.07,
9.14,
9.26,
12.802922887054116,
11.731647935221925
]

SNR_uten_HF_basis_gronn_refl = [
7.793452455399117,
6.7444053625539615,
7.712146551859068,
7.57,
5.3,
7.45,
6.75,
8.030506350140925
]

SNR_uten_HF_basis_blaa_refl = [4.46,
6.35,
5.2,
6.39,
6.33,
3.64,
3.98,
5.24]

basis_refl = [SNR_uten_HF_basis_rod_refl, SNR_uten_HF_basis_gronn_refl, SNR_uten_HF_basis_blaa_refl]
#------------------------------------------------------------------------------------------------------------------------------
#Basismålinger SNR transmittans
SNR_uten_HF_basis_rod_tran = [7.292490597399164,
5.797534644700095,
4.55,
6.972088419340271,
7.387802848760106,
4.430198131216714,
6.105332613954325,
5.740788518785413]

SNR_uten_HF_basis_gronn_tran = [9.771220048499616,
7.9626609187989335,
6.4868609355126114,
8.832393551915603,
9.323503961627484,
7.008051482914356,
9.100398243007858,
7.44835517567093]

SNR_uten_HF_basis_blaa_tran = [5.519433616505074,
1.93,
2.92,
2.88,
2.35,
1.79,
3.8,
1.59
]

basis_tran = [SNR_uten_HF_basis_rod_tran, SNR_uten_HF_basis_gronn_tran, SNR_uten_HF_basis_blaa_tran]
#------------------------------------------------------------------------------------------------------------------------------
#Robusthetstest SNR kaldfinger transmittans
SNR_uten_HF_kaldfinger_rod = [
6.840880485233381,
6.6704585695256,
3.42,
4.66,
5.07,
4.92,
6.841534724634847,
5.331117974340463]

SNR_uten_HF_kaldfinger_gronn = [
8.944600328420178,
6.63,
6.33,
4.26,
3.94,
8.06,
5.19,
3.46
]

SNR_uten_HF_kaldfinger_blaa = [
3.029916212190853,
3.69,
3.06,
2.78,
2.28,
1.86,
2.74,
2.67
]

kaldfinger = [SNR_uten_HF_kaldfinger_rod, SNR_uten_HF_kaldfinger_gronn, SNR_uten_HF_kaldfinger_blaa]
#------------------------------------------------------------------------------------------------------------------------------
#Robusthetstest SNR varmfinger transmittans
SNR_uten_HF_varmfinger_rod = [7.667564128713557,
4.938447677069073,
5.540927280345661,
7.273975102280138,
3.87,
5.268441003875872,
7.802320604981622,
4.855450465517483]

SNR_uten_HF_varmfinger_gronn = [9.862806656303553,
6.56658900227626,
6.605120112891029,
7.356622903843735,
5.331832234446646,
7.381214147207085,
7.825395261723643,
5.919538406478454]

SNR_uten_HF_varmfinger_blaa = [4.912586889468762,
2.4,
3,
3.48,
3.9500294587915414,
4.067882338704639,
3.366266507489606,
3.5453523581743607]

varmfinger = [SNR_uten_HF_varmfinger_rod, SNR_uten_HF_varmfinger_gronn, SNR_uten_HF_varmfinger_blaa]

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
df.to_csv("Lab3/SNR_beregnigner_uten_preprosessering_med_korrigering.csv",index=False)
#------------------------------------------------------------------------------------------------------------------------------
#skriving av SNR_beregninger til terminal
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